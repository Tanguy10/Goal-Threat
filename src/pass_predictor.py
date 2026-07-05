import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import os
import joblib
from config import MODELS_DIR

class PassPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.optimal_params = None
        self.model_loaded = False

    def load_model(self):
        """Load the best trained model with its optimal parameters"""

        # Load the optimal parameters first
        best_params_path = MODELS_DIR / "best_params.json"
        if best_params_path.exists():
            try:
                with open(best_params_path, "r") as f:
                    self.optimal_params = json.load(f)
                print(f"Paramètres optimaux chargés: {self.optimal_params}")
            except Exception as e:
                print(f"Erreur lors du chargement de best_params.json: {e}")
                self.optimal_params = None
        else:
            print("Attention: best_params.json non trouvé")
            self.optimal_params = None

        # Detect and load the best model
        model_files = [
            ("best_pass_model_rf.pkl", "RandomForest", "pkl"),
            ("best_pass_model_lr.pkl", "LogisticRegression", "pkl"),
            ("best_pass_model_xgb.pkl", "XGBoost_sklearn", "pkl"),
            ("best_pass_model_xgb.json", "XGBoost_native", "json"),
        ]

        model_loaded = False

        for filename, model_name, file_type in model_files:
            filename = str(MODELS_DIR / filename)
            if os.path.exists(filename):
                try:
                    if file_type == "pkl":
                        # scikit-learn or XGBClassifier model saved with joblib/pickle
                        with open(filename, 'rb') as f:
                            self.model = pickle.load(f)
                        print(f"✅ Modèle chargé: {model_name} depuis {filename}")
                        model_loaded = True
                        break
                    elif file_type == "json":
                        # Native XGBoost model (Booster)
                        self.model = xgb.Booster()
                        self.model.load_model(filename)
                        print(f"✅ Modèle XGBoost natif chargé depuis {filename}")
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"❌ Erreur chargement {filename}: {e}")
                    continue

        if not model_loaded:
            raise FileNotFoundError("Aucun modèle best_pass_model_* trouvé (pkl ou json)")

        # Load the metadata
        metadata_path = MODELS_DIR / "pass_model_metadata.pkl"
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                self.scaler = metadata.get("scaler")
                self.features = metadata.get("features")
                print("✅ Métadonnées chargées avec succès")
                if self.features:
                    print(f"Features utilisées: {self.features}")
            except Exception as e:
                print(f"Erreur chargement métadonnées: {e}")
                self.features = [
                    'distance_passe', 'sens_passe',
                    'nb_adv_proches_depart', 'nb_adv_trajectoire',
                    'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee'
                ]
                print(f"Utilisation des features par défaut: {self.features}")
        else:
            print("pass_model_metadata.pkl non trouvé")
            self.features = [
                'distance_passe', 'sens_passe',
                'nb_adv_proches_depart', 'nb_adv_trajectoire',
                'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee'
            ]

        self.model_loaded = True
        return self
    
    def predict_proba(self, df_input):
        """
        Required method for compatibility with scikit-learn and notebooks.
        Returns an array of shape (n_samples, 2) = [[P(neg), P(pos)], ...]
        """
        probs = self.predict_pass_success(df_input)
        # Transform into format [[1-p, p], [1-p, p], ...] for each sample
        return np.vstack([1 - probs, probs]).T

    def get_optimal_params(self):
        """Return the optimal parameters found by Optuna"""
        if not self.model_loaded:
            self.load_model()
        return self.optimal_params

    def predict_pass_success(self, df_input):
        """
        Predict the pass success probability with the optimal model
        
        Args:
            df_input: DataFrame with the columns matching the features
            
        Returns:
            Array of success probabilities between 0 and 1
        """
        if not self.model_loaded:
            self.load_model()

        # Check that all required columns are present
        missing_cols = [col for col in self.features if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")

        # Scale the features if a scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(df_input[self.features])
        else:
            X_scaled = df_input[self.features].values
            print("Attention: pas de scaler disponible, utilisation des données brutes")
        
        # Predict the probabilities
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]
        elif isinstance(self.model, xgb.Booster):
            dmat = xgb.DMatrix(X_scaled, feature_names=self.features)
            return self.model.predict(dmat)
        else:
            return self.model.predict(X_scaled)

    def predict_single_pass(self, x_passeur, y_passeur, x_cible, y_cible,
                          densite_adv_depart=1.0, nb_adv_trajectoire=1,
                          densite_adv_arrivee=1.0, densite_coequipiers_arrivee=1.0):
        """
        Predict the success probability for a specific pass with the optimal parameters
        
        Args:
            x_passeur, y_passeur: Passer position (normalized coordinates 0-1)
            x_cible, y_cible: Target position (normalized coordinates 0-1)
            densite_adv_depart: Density of opponents near the passer (continuous value)
            nb_adv_trajectoire: Number of opponents on the trajectory (integer)
            densite_adv_arrivee: Density of opponents near the target (continuous value)
            densite_coequipiers_arrivee: Density of teammates near the target (continuous value)
            
        Returns:
            Success probability (0-1)
        """
                # Compute distance_passe and sens_passe as in Pass_chances_function

        # Create a DataFrame with the right features
        pass_data = pd.DataFrame({
            'x_passeur': [x_passeur],
            'y_passeur': [y_passeur],
            'nb_adv_proches_depart': [densite_adv_depart],  # Density, not count
            'nb_adv_trajectoire': [nb_adv_trajectoire],      # Count
            'nb_adv_proches_arrivee': [densite_adv_arrivee], # Density, not count
            'nb_coequipiers_proches_arrivee': [densite_coequipiers_arrivee] # Density
        })
        
        return self.predict_pass_success(pass_data)[0]

    def print_model_info(self):
        """Print information about the model and its optimal parameters"""
        if not self.model_loaded:
            self.load_model()
            
        print("=== INFORMATIONS DU MODÈLE OPTIMAL ===")
        print(f"Type de modèle: {type(self.model).__name__}")
        print(f"Features utilisées: {self.features}")
        
        if self.optimal_params:
            print("Paramètres optimaux trouvés par Optuna:")
            for param, value in self.optimal_params.items():
                print(f"  {param}: {value}")
        else:
            print("Paramètres optimaux non disponibles")

    def calculate_pass_features(self, x_passeur, y_passeur, x_cible, y_cible, 
                               teammates_pos, opponents_pos):
        """
        Compute all the features needed for a given pass
        
        Args:
            x_passeur, y_passeur: Passer position (normalized 0-1)
            x_cible, y_cible: Target position (normalized 0-1)
            teammates_pos: Array of teammate positions (normalized)
            opponents_pos: Array of opponent positions (normalized)
            
        Returns:
            dict: Dictionary with all the features
        """
        # Default parameters (adjustable according to the optimal parameters)
        if self.optimal_params:
            sigma = self.optimal_params.get('sigma', 0.05)
            seuil_trajectoire = self.optimal_params.get('seuil_trajectoire', 0.02)
        else:
            print("Aucun paramètre optimal trouvé, utilisation des valeurs par défaut")
            sigma = 0.05
            seuil_trajectoire = 0.02
            
        # Pass distance and direction
        distance_passe = np.sqrt((x_cible - x_passeur)**2 + (y_cible - y_passeur)**2)
        sens_passe = 1 if x_cible > x_passeur else -1
        
        # Passer and target positions
        passeur_pos = np.array([x_passeur, y_passeur])
        cible_pos = np.array([x_cible, y_cible])
        
        # Compute opponents near the start (Gaussian density)
        if len(opponents_pos) > 0:
            distances_depart = np.linalg.norm(opponents_pos - passeur_pos, axis=1)
            densites_depart = np.exp(-(distances_depart**2) / (2 * sigma**2))
            nb_adv_proches_depart = np.sum(densites_depart)
        else:
            nb_adv_proches_depart = 0
            
        # Compute opponents on the trajectory
        nb_adv_trajectoire = self._count_opponents_on_trajectory(
            passeur_pos, cible_pos, opponents_pos, seuil_trajectoire
        )
        
        # Compute opponents near the target (Gaussian density)
        if len(opponents_pos) > 0:
            distances_arrivee = np.linalg.norm(opponents_pos - cible_pos, axis=1)
            densites_arrivee = np.exp(-(distances_arrivee**2) / (2 * sigma**2))
            nb_adv_proches_arrivee = np.sum(densites_arrivee)
        else:
            nb_adv_proches_arrivee = 0
            
        # Compute teammates near the target (Gaussian density)
        if len(teammates_pos) > 0:
            distances_coequipiers = np.linalg.norm(teammates_pos - cible_pos, axis=1)
            densites_coequipiers = np.exp(-(distances_coequipiers**2) / (2 * sigma**2))
            nb_coequipiers_proches_arrivee = np.sum(densites_coequipiers)
        else:
            nb_coequipiers_proches_arrivee = 0
        
        return {
            'distance_passe': distance_passe,
            'sens_passe': sens_passe,
            'nb_adv_proches_depart': nb_adv_proches_depart,
            'nb_adv_trajectoire': nb_adv_trajectoire,
            'nb_adv_proches_arrivee': nb_adv_proches_arrivee,
            'nb_coequipiers_proches_arrivee': nb_coequipiers_proches_arrivee
        }
    
    def _count_opponents_on_trajectory(self, passeur_pos, cible_pos, opponents_pos, seuil_trajectoire):
        """Count the opponents on the pass trajectory"""
        if len(opponents_pos) == 0:
            return 0
            
        traj_vect = cible_pos - passeur_pos
        norm_traj = np.linalg.norm(traj_vect)
        
        if norm_traj == 0:
            return 0
            
        traj_vect_norm = traj_vect / norm_traj
        
        # Compute the projections
        adv_vectors = opponents_pos - passeur_pos
        projections = np.dot(adv_vectors, traj_vect_norm)
        
        # Opponents on the trajectory
        on_trajectory = (projections > 0) & (projections < norm_traj)
        
        if not np.any(on_trajectory):
            return 0
            
        # Distance to the trajectory line
        points_on_line = passeur_pos + np.outer(projections[on_trajectory], traj_vect_norm)
        distances = np.linalg.norm(opponents_pos[on_trajectory] - points_on_line, axis=1)
        
        return np.sum(distances < seuil_trajectoire)
    
    def predict_pass_probability(self, features_df):
        """
        Predict the success probability of a pass
        
        Args:
            features_df: DataFrame with the features
            
        Returns:
            array: Success probabilities
        """
        if not self.model_loaded:
            self.load_model()
            
        # Check that all features are present
        for feature in self.features:
            if feature not in features_df.columns:
                raise ValueError(f"Feature manquante: {feature}")
        
        # Select and order the features
        X = features_df[self.features]
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities



# Example usage
if __name__ == "__main__":
    predictor = PassPredictor().load_model()
    predictor.print_model_info()
    
    # Test a prediction with the correct parameter names
    prob = predictor.predict_single_pass(
        x_passeur=0.3, y_passeur=0.4,  # Normalized coordinates
        x_cible=0.7, y_cible=0.2,
        densite_adv_depart=2.0,        # Density of opponents at the start
        nb_adv_trajectoire=1,          # Number of opponents on the trajectory
        densite_adv_arrivee=1.0,       # Density of opponents at the target
        densite_coequipiers_arrivee=2.0 # Density of teammates at the target
    )
    print(f"\nProbabilité de succès de la passe: {prob:.1%}")