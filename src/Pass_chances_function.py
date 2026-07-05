import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statsbombpy import sb
import pickle
import optuna
from config import MODELS_DIR, STATSBOMB_360_DIR

def objective_features(trial):
    """Objective function to optimize the feature hyperparameters"""
    
    # Suggest values for each hyperparameter
    sigma = trial.suggest_float('sigma', 0.01, 0.1)
    seuil_trajectoire = trial.suggest_float('seuil_trajectoire', 0.005, 0.05)

    
    try:
        # Recreate the data with these parameters
        data = create_passing_database(
            sigma = sigma,
            seuil_trajectoire=seuil_trajectoire,
            max_matches=50  # Sample to speed things up
        )
        
        # Train the model quickly
        X = data[['x_passeur', 'y_passeur', 'x_cible', 'y_cible', 
                 'nb_adv_proches_depart', 'nb_adv_trajectoire',
                 'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee']]
        y = data['succès']
        
        # Quick cross-validation
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
        
        return scores.mean()
        
    except Exception as e:
        print(f"Erreur avec paramètres: {e}")
        return 0.5  # Neutral score if error

def densite_adversaires_ponderee(x_cible, y_cible, adv_positions, sigma=0.05):
    """
    Compute a density of opponents with Gaussian decay
    """
    if len(adv_positions) == 0:
        return 0
    
    cible = np.array([x_cible, y_cible])
    distances = np.linalg.norm(adv_positions - cible, axis=1)
    
    # Gaussian function: the closer, the larger the weight
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    
    return np.sum(weights)

def densite_adversaires_inverse(x_cible, y_cible, adv_positions, alpha=1.0):
    """
    Density with inverse-power decay
    """
    if len(adv_positions) == 0:
        return 0
    
    cible = np.array([x_cible, y_cible])
    distances = np.linalg.norm(adv_positions - cible, axis=1)
    
    # Avoid division by zero
    distances = np.maximum(distances, 0.01)
    
    # Weight inversely proportional to the distance
    weights = 1 / (distances**alpha)
    
    return np.sum(weights)

def get_zone(x, y, x_divisions=15, y_divisions=10):
    """
    Convert (x,y) coordinates to zone number using actual pitch dimensions
    
    Args:
        x: x-coordinate (0-1 normalized)
        y: y-coordinate (0-1 normalized)
        x_divisions: number of divisions along x-axis (width of pitch)
        y_divisions: number of divisions along y-axis (height of pitch)
        
    Returns:
        Zone number (1-based, row-major order)
    """
    # Scale back to actual pitch dimensions
    x_actual = x * 120
    y_actual = y * 80
    
    # Calculate zone indices
    x_idx = min(int(x_actual / (120 / x_divisions)), x_divisions - 1)
    y_idx = min(int(y_actual / (80 / y_divisions)), y_divisions - 1)
    
    # Calculate zone number (1-based indexing)
    return x_idx * y_divisions + y_idx + 1

def diff_distance_joueurs_proches(x_cible, y_cible, adv_positions, teammate_positions, x_passeur, y_passeur):
    """
    Compute the difference between the distance of the closest defender and that of the closest
    teammate, divided by the pass length.
    
    A positive value means the teammate is closer than the defender (favorable situation).
    A negative value means the defender is closer (unfavorable situation).
    
    Args:
        x_cible, y_cible: Coordinates of the pass reception point
        adv_positions: Array of opponent coordinates
        teammate_positions: Array of teammate coordinates
        x_passeur, y_passeur: Coordinates of the pass starting point
        
    Returns:
        float: Normalized distance difference
    """
    # Pass distance (for normalization)
    pass_distance = np.sqrt((x_cible - x_passeur)**2 + (y_cible - y_passeur)**2)
    if pass_distance == 0:
        return 0  # Avoid division by zero
    
    # Target point
    cible = np.array([x_cible, y_cible])
    
    # Find the closest defender
    if len(adv_positions) > 0:
        adv_distances = np.linalg.norm(adv_positions - cible, axis=1)
        min_adv_distance = np.min(adv_distances)
    else:
        print("⚠️ Aucun adversaire trouvé")
        min_adv_distance = 1.0  # Default value if no opponent
    
    # Find the closest teammate
    if len(teammate_positions) > 0:
        teammate_distances = np.linalg.norm(teammate_positions - cible, axis=1)
        min_teammate_distance = np.min(teammate_distances)
    else:
        print("⚠️ Aucun coéquipier trouvé")
        min_teammate_distance = 1.0  # Default value if no teammate
    
    # Compute the normalized difference
    # Positive: teammate closer, Negative: defender closer
    return (min_adv_distance - min_teammate_distance) / pass_distance

# Utility function: Euclidean distance computation
def distance(p1_x, p1_y, p2_x, p2_y):
    return np.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2)

def nb_adv_proches_coords(x_passeur, y_passeur, adv_positions, seuil_proche=0.05):
    """Improved version that accepts coordinates directly"""
    passeur = np.array([x_passeur, y_passeur])
    distances = np.linalg.norm(adv_positions - passeur, axis=1)
    return np.sum(distances < seuil_proche)

def nb_adv_trajectoire_coords(x_passeur, y_passeur, x_cible, y_cible, adv_positions, seuil_trajectoire=0.02):
    """Improved version that accepts coordinates directly"""
    passeur = np.array([x_passeur, y_passeur])
    cible = np.array([x_cible, y_cible])
    
    traj_vect = cible - passeur
    
    # Avoid division by zero
    norm_traj = np.linalg.norm(traj_vect)
    if norm_traj == 0:
        return 0
        
    traj_vect_norm = traj_vect / norm_traj
    
    # Calculate projections and distances
    adv_vectors = adv_positions - passeur
    projections = np.dot(adv_vectors, traj_vect_norm)
    
    # Find opponents on the trajectory
    on_trajectory = (projections > 0) & (projections < norm_traj)
    
    if not np.any(on_trajectory):
        return 0
        
    # For each opponent on trajectory, compute distance to the line
    points_on_line = passeur + np.outer(projections[on_trajectory], traj_vect_norm)
    distances = np.linalg.norm(adv_positions[on_trajectory] - points_on_line, axis=1)
    
    return np.sum(distances < seuil_trajectoire)

def nb_coequipiers_trajectoire(row, team_positions, seuil_distance_traj=0.02):
    """Count teammates on the pass trajectory"""
    # This function is identical to nb_adv_trajectoire but used for teammates
    passeur = np.array([row['x_passeur'], row['y_passeur']])
    cible = np.array([row['x_cible'], row['y_cible']])
    traj_vect = cible - passeur
    
    norm_traj = np.linalg.norm(traj_vect)
    if norm_traj == 0:
        return 0
        
    traj_vect_norm = traj_vect / norm_traj
    
    team_vectors = team_positions - passeur
    projections = np.dot(team_vectors, traj_vect_norm)
    
    on_trajectory = (projections > 0) & (projections < norm_traj)
    
    if not np.any(on_trajectory):
        return 0
        
    points_on_line = passeur + np.outer(projections[on_trajectory], traj_vect_norm)
    distances = np.linalg.norm(team_positions[on_trajectory] - points_on_line, axis=1)
    
    return np.sum(distances < seuil_distance_traj)

# Main function to run the analysis
def analyze_passing_success(data):
    # Load the data
    try:
        print("Attempting to load StatsBomb 360 data...")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data for testing purposes...")
        # Create dummy data for testing if loading fails
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'x_passeur': np.random.uniform(0, 1, n_samples),
            'y_passeur': np.random.uniform(0, 1, n_samples),
            'x_cible': np.random.uniform(0, 1, n_samples),
            'y_cible': np.random.uniform(0, 1, n_samples),
            'distance_passe': np.random.uniform(0.05, 0.5, n_samples),
            'nb_adv_proches': np.random.randint(0, 6, n_samples),
            'nb_adv_trajectoire': np.random.randint(0, 4, n_samples),
        })
        
        # Generate success probability based on features
        probs = 1 / (1 + np.exp(3 * data['nb_adv_trajectoire'] + 
                                 data['nb_adv_proches'] + 
                                 5 * data['distance_passe'] - 2))
        data['succès'] = (np.random.random(n_samples) < probs).astype(int)
    
    # Basic data exploration
    print("\n=== Pass Success Data Overview ===")
    print(f"Total passes: {len(data)}")
    print(f"Success rate: {data['succès'].mean():.2%}")
    
    print("\nFeature distributions:")
    data.describe().round(3).T.to_string()
    
    # Feature engineering
    print("\n=== Feature Engineering ===")

    # Feature and label selection
    features = ['x_passeur', 'y_passeur', 'x_cible', 'y_cible',
            'nb_adv_proches_depart', 'nb_adv_trajectoire',
            'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee' ,'diff_distance_normalisee' # New features
            ]
    
    # Make sure all features exist in data
    features = [f for f in features if f in data.columns]
    
    X = data[features]
    y = data['succès']
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Data split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    
    print(f"\n=== Model Training ===")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Models to test
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    
    # Store results for comparison
    results = {}
    
    # Training and evaluation
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]
        
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC: {auc:.3f}")
        
        results[name] = {
            'model': model,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_model = results[best_model_name]['model']
    print(f"\n🏆 Best model: {best_model_name} (AUC: {results[best_model_name]['auc']:.3f})")
    
    # Create and display the pass success probability function
    def pass_success_probability(df_input):
        """Function that evaluates pass success probability given input features"""
        # Ensure input has all required features
        required_cols = features
        for col in required_cols:
            if col not in df_input.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Scale the input
        X_input = scaler.transform(df_input[features])
        
        # Predict probabilities
        return best_model.predict_proba(X_input)[:, 1]
    
    # Inside the analyze_passing_success function, modify the model saving section:
    
    # Save the best model
    print("\n=== Saving Models ===")
    # Save all models, not just the best one
    for model_name, result in results.items():
        model = result['model']
        print(f"Saving {model_name} model...")
        
        if model_name == "XGBoost":
            model_path = str(MODELS_DIR / f"pass_model_{model_name.lower()}.json")
            model.save_model(model_path)
        else:
            import pickle
            model_path = str(MODELS_DIR / f"pass_model_{model_name.lower()}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        print(f"{model_name} model saved to {model_path}")

    # Also save the best model with a special name for easy reference
    if best_model_name == "XGBoost":
        model_path = str(MODELS_DIR / "best_pass_model_xgb.json")
        best_model.save_model(model_path)
    else:
        import pickle
        model_path = str(MODELS_DIR / "best_pass_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

    print(f"Best model saved to {model_path}")

    import pickle
    # Save scaler and features
    metadata_path = str(MODELS_DIR / "pass_model_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump({"scaler": scaler, "features": features}, f)
    print(f"Model metadata (scaler and features) saved to {metadata_path}")
    return pass_success_probability, data, best_model, features, scaler

def create_passing_database(three_sixty_folder=str(STATSBOMB_360_DIR), match_ids=None, max_matches=None,
                           x_divisions=15, y_divisions=10, sigma=0.05, seuil_trajectoire=0.02):
    """
    Create a database of passes with features for analysis
    
    Args:
        three_sixty_folder: Path to the folder containing StatsBomb 360 JSON files
        match_ids: List of specific match IDs to load, or None to load based on max_matches
        max_matches: Maximum number of matches to load if match_ids is None
        x_divisions: Number of zones along the x-axis of the field
        y_divisions: Number of zones along the y-axis of the field
        seuil_proche: Distance threshold to consider a player "close" (in normalized units)
        seuil_trajectoire: Distance threshold to consider a player on trajectory (in normalized units)
        
    Returns:
        DataFrame with processed pass data including all requested features
    """
    print("📊 Creating passing database from StatsBomb 360 data...")
    
    # Path to the 360 data folder
    three_sixty_folder = Path(three_sixty_folder)
    
    # Verify the folder exists
    if not three_sixty_folder.exists():
        raise FileNotFoundError(f"Le dossier {three_sixty_folder} n'existe pas")
    
    # Get JSON files based on input parameters
    if match_ids:
        # Convert to list if single ID provided
        if isinstance(match_ids, (int, str)):
            match_ids = [str(match_ids)]
        else:
            match_ids = [str(mid) for mid in match_ids]
            
        json_files = [three_sixty_folder / f"{match_id}.json" for match_id in match_ids 
                     if (three_sixty_folder / f"{match_id}.json").exists()]
        
        if not json_files:
            raise ValueError(f"Aucun des IDs de match fournis n'a été trouvé dans {three_sixty_folder}")
            
    else:
        all_json_files = list(three_sixty_folder.glob("*.json"))
        if not all_json_files:
            raise ValueError(f"Aucun fichier JSON trouvé dans {three_sixty_folder}")
            
        if max_matches and max_matches < len(all_json_files):
            # Randomly select max_matches files
            import random
            random.seed(42)  # For reproducibility
            json_files = random.sample(all_json_files, max_matches)
        else:
            json_files = all_json_files
    
    print(f"Traitement de {len(json_files)} fichiers de match")
    
    all_passes = []
    total_passes_found = 0
    passes_with_360_data = 0
    successfully_processed = 0
    
    # Process each match file
    for file_path in tqdm(json_files, desc="Traitement des matchs"):
        current_match_id = file_path.stem
        try:
            # Load the 360 data
            with open(file_path, 'r', encoding='utf-8') as f:
                data_360 = json.load(f)
            print(f"Chargement des données 360 pour le match {current_match_id}...", flush = True)
            # Create mapping from event ID to 360 data
            event_360_map = {e["event_uuid"]: e for e in data_360 if "event_uuid" in e}
            if not event_360_map:
                print(f"⚠️ Le match {current_match_id} ne contient pas de données 360 valides, ignoré.")
                continue
                
            # Load events data using statsbombpy
            try:
                events = sb.events(match_id=int(current_match_id))
                print(f"Chargement des événements pour le match {current_match_id}...")
                # Filter for pass events with complete data
                if 'pass_end_location' in events.columns:
                    pass_events = events[
                        (events['type'].apply(lambda x: x.get('name') if isinstance(x, dict) else x) == 'Pass') & 
                        (events['pass_end_location'].notna())  # Use pass_end_location instead of pass
                    ]
                else:
                    print(f"⚠️ Le match {current_match_id} ne contient pas de colonne 'pass_end_location', ignoré.")
                    continue
                
                total_passes_found += len(pass_events)
                
                if len(pass_events) == 0:
                    print(f"⚠️ Le match {current_match_id} ne contient pas de passes valides, ignoré.")
                    continue
                    
                # Process each pass event
                for _, event in pass_events.iterrows():
                    event_id = event['id']
                    
                    # Skip passes without 360 data
                    if event_id not in event_360_map:
                        continue
                        
                    event_360 = event_360_map[event_id]
                    
                    # Skip events without freeze_frame
                    if 'freeze_frame' not in event_360 or not event_360['freeze_frame']:
                        continue
                        
                    passes_with_360_data += 1
                    
                    # Extract basic pass information
                    pass_info = {
                        'succès': 1 if pd.isna(events.loc[_, 'pass_outcome']) else 0,  # pass_outcome instead of pass.outcome
                        'x_passeur': event['location'][0] / 120,
                        'y_passeur': event['location'][1] / 80,
                        'x_cible': event['pass_end_location'][0] / 120,  # Direct access to pass_end_location
                        'y_cible': event['pass_end_location'][1] / 80
                    }
                

                    # pass_info['sens_passe'] = 1 if pass_info['x_cible'] > pass_info['x_passeur'] else -1
                    # # Calculate zones
                    pass_info['zone_depart'] = get_zone(event['location'][0] / 120, event['location'][1] / 80, x_divisions, y_divisions)
                    pass_info['zone_arrivee'] = get_zone(event['pass_end_location'][0] / 120, event['pass_end_location'][1] / 80, x_divisions, y_divisions)

                    # Extract player positions from freeze_frame
                    freeze_frame = event_360.get('freeze_frame', [])
                    teammates = []
                    opponents = []
                    
                    for player in freeze_frame:
                        if not player.get('location'):
                            continue
                            
                        player_pos = [player['location'][0] / 120, player['location'][1] / 80]
                        
                        if player.get('teammate'):
                            teammates.append(player_pos)
                        else:
                            opponents.append(player_pos)
                        
                    # Calculate features based on opponents
                    if opponents:
                        adv_positions = np.array(opponents)
    
                        try:
                            # Opponents near pass origin
                            pass_info['nb_adv_proches_depart'] = densite_adversaires_ponderee(
                            pass_info['x_passeur'], pass_info['y_passeur'], 
                            adv_positions, sigma
                            )         
        
                            # Opponents on trajectory
                            pass_info['nb_adv_trajectoire'] = nb_adv_trajectoire_coords(
                                pass_info['x_passeur'], pass_info['y_passeur'],
                                pass_info['x_cible'], pass_info['y_cible'],
                                adv_positions, seuil_trajectoire
                            )
                            pass_info['nb_adv_proches_arrivee'] = densite_adversaires_ponderee(
                                pass_info['x_cible'], pass_info['y_cible'],
                                adv_positions, sigma
                            )
                        except Exception as e:
                            print(f"Erreur détaillée: {str(e)}")
                            # Default values in case of error
                            pass_info['nb_adv_proches_depart'] = 0
                            pass_info['nb_adv_trajectoire'] = 0
                            pass_info['nb_adv_proches_arrivee'] = 0
                    else:
                        pass_info['nb_adv_proches_depart'] = 0
                        pass_info['nb_adv_trajectoire'] = 0
                        pass_info['nb_adv_proches_arrivee'] = 0                   
                    
                    # Calculate features based on teammates
                    if teammates:
                        teammates = [coeq for coeq in teammates if distance(coeq[0],coeq[1], pass_info['x_passeur'], pass_info['y_passeur']) > 0.005]
                        team_positions = np.array(teammates)
                        
                        # Teammates on trajectory
                        pass_info['nb_coequipiers_trajectoire'] = nb_adv_trajectoire_coords(
                            pass_info['x_passeur'], pass_info['y_passeur'],
                            pass_info['x_cible'], pass_info['y_cible'],
                            team_positions, sigma
                        )
                        pass_info['nb_coequipiers_proches_arrivee'] = densite_adversaires_ponderee(
                            pass_info['x_cible'], pass_info['y_cible'],
                            team_positions, sigma
                        )
                    else:
                        pass_info['nb_coequipiers_trajectoire'] = 0
                        pass_info['nb_coequipiers_proches_arrivee'] = 0
                    pass_info['diff_distance_normalisee'] = diff_distance_joueurs_proches(
                        pass_info['x_cible'], pass_info['y_cible'],
                        adv_positions, team_positions,
                        pass_info['x_passeur'], pass_info['y_passeur']
                    )
                    all_passes.append(pass_info)
                    successfully_processed += 1
                    
            except Exception as e:
                print(f"❌ Erreur lors du traitement des événements pour le match {current_match_id}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement des données 360 pour le match {current_match_id}: {str(e)}")
            continue
    
    # Create DataFrame from collected data
    if not all_passes:
        raise ValueError("Aucune passe valide n'a été trouvée pour créer la base de données")
    
    df = pd.DataFrame(all_passes)
    
    # Print summary statistics
    print("\n=== Statistiques de création de la base de données ===")
    print(f"Passes trouvées: {total_passes_found}")
    print(f"Passes avec données 360: {passes_with_360_data}")
    print(f"Passes traitées avec succès: {successfully_processed}")
    print(f"Base de données finale: {len(df)} passes de {len(json_files)} matchs")
    
    return df

# Run the analysis if this is the main script
if __name__ == "__main__":
    # Choose what to do
    optimize_hyperparams = False  # Set to False to skip the optimization
    
    if optimize_hyperparams:
        print("🔍 Optimisation des hyperparamètres en cours...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_features, n_trials=50)
        
        print("Meilleurs hyperparamètres de features:")
        print(study.best_params)
        print(f"Meilleur score: {study.best_value:.3f}")
        
        # Save the optimization results
        import pickle
        with open(str(MODELS_DIR / "hyperparameter_optimization_results.pkl"), "wb") as f:
            pickle.dump({
                'study': study,
                'best_params': study.best_params,
                'best_value': study.best_value
            }, f)
        print("Résultats d'optimisation sauvegardés")

        print("Étude terminée!")
        print(f"Meilleur score: {study.best_value}")
        print(f"Meilleurs paramètres: {study.best_params}")
        import json
        with open(str(MODELS_DIR / 'best_params.json'), 'w') as f:
            json.dump(study.best_params, f, indent=2)

        print("Paramètres sauvegardés dans 'best_params.json'")
        
        
        print("Pour utiliser les hyperparamètres optimisés, relancez le script avec optimize_hyperparams=False")
        print("et modifiez manuellement les paramètres dans create_passing_database()")
    else:
        # Load the optimal parameters from the JSON file
        import json
        import os
        
        if os.path.exists(str(MODELS_DIR / 'best_params.json')):
            print("Chargement des paramètres optimaux depuis best_params.json...")
            with open(str(MODELS_DIR / 'best_params.json'), 'r') as f:
                best_params = json.load(f)
            
            print(f"Paramètres chargés: {best_params}")
            
            # Create the final database with the optimal parameters
            print("Création de la base de données finale avec les paramètres optimaux...")
            final_data = create_passing_database(
                sigma=best_params['sigma'],
                seuil_trajectoire=best_params['seuil_trajectoire'],
                max_matches=None  # Use all matches
            )

            # Now use analyze_passing_success with the optimal data
            print("Analyse finale avec les données optimisées...")
            final_model = analyze_passing_success(final_data)

            print("Analyse finale terminée avec les paramètres optimisés par Optuna!")
        else:
            print("Fichier best_params.json introuvable. Lancez d'abord l'optimisation avec optimize_hyperparams=True")
        