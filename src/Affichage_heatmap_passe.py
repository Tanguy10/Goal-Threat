import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import pickle
from config import METRICA_DIR, PROCESSED_DIR, MODELS_DIR, HEATMAPS_DIR

df_filtered = pd.read_csv(str(PROCESSED_DIR / "df_filtered.csv"))
with open(str(PROCESSED_DIR / "liste_goal_threat.pkl"), "rb") as f:
    liste_goal_threat = pickle.load(f)

csv_home = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Home_Team.csv")
csv_away = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Away_Team.csv")


from Pass_chances_function import (
    distance, nb_adv_proches_coords, nb_adv_trajectoire_coords,
    nb_coequipiers_trajectoire, densite_adversaires_ponderee,
    densite_adversaires_inverse, get_zone, diff_distance_joueurs_proches
)

def create_pitch(ax):
    """Create a standardized football pitch (normalized proportions)"""
    # Pitch (green rectangle)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                           edgecolor='white', facecolor='#538032', alpha=0.8)
    ax.add_patch(rect)
    
    # Halfway line
    plt.plot([0.5, 0.5], [0, 1], color='white', linewidth=2)
    
    # Center circle
    circle = plt.Circle((0.5, 0.5), 0.1, color='white', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Penalty area (left)
    rect = patches.Rectangle((0, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Goal area (left)
    rect = patches.Rectangle((0, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Penalty area (right)
    rect = patches.Rectangle((0.82, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Goal area (right)
    rect = patches.Rectangle((0.94, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Adjust the plot parameters
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def plot_pass_success_probability(ax, origin_x, origin_y, target_x, target_y, prob, annotate=True):
    """Draw a pass and its success probability"""
    # Draw the pass (arrow)
    arrow = ax.arrow(origin_x, origin_y, target_x-origin_x, target_y-origin_y, 
                   head_width=0.02, head_length=0.03, fc='black', ec='black',
                   length_includes_head=True, alpha=0.7)
    
    # Display the probability
    if annotate:
        mid_x = (origin_x + target_x) / 2
        mid_y = (origin_y + target_y) / 2
        ax.annotate(f"{prob:.0%}", (mid_x, mid_y), 
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                  fontsize=10, ha='center', va='center')
    
    return arrow

def plot_player(ax, x, y, team='home', label=None):
    """Draw a player on the pitch"""
    if team == 'home':
        color = 'blue'
    elif team == 'away':
        color = 'red'
    else:
        color = 'gray'
    
    circle = ax.scatter(x, y, s=200, color=color, edgecolor='black', zorder=3, 
                      alpha=0.8, label=label)
    
    if label:
        ax.annotate(label, (x, y), fontsize=8, ha='center', va='center',
                  color='white', fontweight='bold', zorder=4)
    
    return circle

def load_best_hyperparameters():
    """Load the optimized hyperparameters"""
    # If the predictor is already loaded, use its parameters
    if 'predictor' in globals() and hasattr(predictor, 'get_optimal_params'):
        params = predictor.get_optimal_params()
        if params:
            print("✅ Hyperparamètres chargés depuis le prédicteur")
            return params
    
    # Otherwise, try to load from the file
    try:
        with open(str(MODELS_DIR / "hyperparameter_optimization_results.pkl"), 'rb') as f:
            optuna_results = pickle.load(f)
        
        best_params = optuna_results['best_params']
        best_value = optuna_results['best_value']
        
        print("✅ Hyperparamètres Optuna chargés avec succès")
        print(f"   Score d'optimisation: {best_value:.4f}")
        print(f"   Paramètres: {best_params}")
        
        return best_params
        
    except FileNotFoundError:
        print("⚠️ Fichier hyperparameter_optimization_results.pkl non trouvé")
        print("   Utilisation des valeurs par défaut")
        return {
            'sigma': 0.05,
            'seuil_trajectoire': 0.02,
            'x_divisions': 15,
            'y_divisions': 10
        }

def print_hyperparameters_info():
    """Print the hyperparameters currently in use"""
    params = load_best_hyperparameters()
    
    print("=" * 45)
    print("    HYPERPARAMÈTRES OPTUNA UTILISÉS")
    print("=" * 45)
    for key, value in params.items():
        print(f"{key:20}: {value}")
    print("=" * 45)

def calculate_pass_features(x_passeur, y_passeur, x_cible, y_cible, 
                          adv_positions, teammate_positions, 
                          x_divisions=15, y_divisions=10):
    """
    Compute all features of a pass using the hyperparameters optimized by Optuna
    """
    # Load Optuna's optimal hyperparameters
    best_params = load_best_hyperparameters()
    
    # Use the optimized values or the default parameters
    sigma_optimal = best_params.get('sigma', 0.05)
    seuil_trajectoire_optimal = best_params.get('seuil_trajectoire', 0.02)
    
    # Use the optimal hyperparameters
    # Opponents near the start (weighted density with optimal sigma)
    nb_adv_proches_depart = densite_adversaires_ponderee(
        x_passeur, y_passeur, adv_positions, 
        sigma=sigma_optimal  # ← Value optimized by Optuna
    )
    
    # Opponents on the trajectory (with optimal threshold)
    nb_adv_trajectoire = nb_adv_trajectoire_coords(
        x_passeur, y_passeur, x_cible, y_cible, adv_positions, 
        seuil_trajectoire=seuil_trajectoire_optimal  # ← Value optimized by Optuna
    )

    # Teammates on the trajectory
    nb_coequipiers_trajectoire = nb_adv_trajectoire_coords(
        x_passeur, y_passeur, x_cible, y_cible, teammate_positions, 
        seuil_trajectoire=seuil_trajectoire_optimal  # ← Value optimized by Optuna
    )
    
    # Opponents near the target (weighted density with optimal sigma)
    nb_adv_proches_arrivee = densite_adversaires_ponderee(
        x_cible, y_cible, adv_positions, 
        sigma=sigma_optimal  # ← Value optimized by Optuna
    )
    
    # Teammates near the target
    nb_coequipiers_proches_arrivee = densite_adversaires_ponderee(
        x_cible, y_cible, teammate_positions, 
        sigma=sigma_optimal  # ← Value optimized by Optuna
    )
    diff_distance = diff_distance_joueurs_proches(
        x_cible, y_cible,
        adv_positions, teammate_positions,
        x_passeur, y_passeur
    )
    # Create the features dictionary (exactly as you defined it)
    features = {
        'x_passeur': x_passeur,
        'y_passeur': y_passeur,
        'x_cible': x_cible,
        'y_cible': y_cible,
        'nb_adv_proches_depart': nb_adv_proches_depart,
        'nb_adv_trajectoire': nb_adv_trajectoire,
        'nb_coequipiers_trajectoire': nb_coequipiers_trajectoire,
        'nb_adv_proches_arrivee': nb_adv_proches_arrivee,
        'nb_coequipiers_proches_arrivee': nb_coequipiers_proches_arrivee,
        'diff_distance_normalisee': diff_distance
    }
    
    return features
# Import the functions from Pass_chances_function.py

def create_situation(situation_name):
    """
    Creates realistic game situations (5v5) with professional-level positioning.
    Players are well distributed in width and depth according to tactical principles.
    """
    situations = {
        'aile_droite': {
            'passeur': (0.65, 0.15),  # Right winger closer to the line
            'adversaires': np.array([
                [0.78, 0.25],  # Fullback
                [0.75, 0.40],  # Center-back
                [0.60, 0.35],  # Defensive midfielder
                [0.70, 0.15],  # Pressing the passer
                [0.50, 0.30]   # Opposite midfielder
            ]),
            'coequipiers': np.array([
                [0.40, 0.50],  # Central midfielder
                [0.55, 0.25],  # Supporting fullback
                [0.75, 0.50],  # Forward
                [0.85, 0.30],  # Forward making a run in behind
                [0.60, 0.60]   # Opposite winger
            ]),
            'title': "Right Wing Attack (5v5)"
        },
        'milieu_terrain': {
            'passeur': (0.50, 0.50),  # Central midfielder
            'adversaires': np.array([
                [0.75, 0.30],  # Right center-back
                [0.75, 0.70],  # Left center-back
                [0.60, 0.40],  # Defensive midfielder
                [0.60, 0.60],  # Box-to-box midfielder
                [0.65, 0.50]   # Withdrawn forward
            ]),
            'coequipiers': np.array([
                [0.30, 0.50],  # Defensive midfielder
                [0.40, 0.20],  # Right fullback
                [0.40, 0.80],  # Left fullback
                [0.70, 0.35],  # Right forward
                [0.70, 0.65]   # Left forward
            ]),
            'title': "Midfield Build-up (5v5)"
        },
        'contre_attaque': {
            'passeur': (0.35, 0.50),  # Ball-winning midfielder
            'adversaires': np.array([
                [0.45, 0.30],  # Recovering defender
                [0.45, 0.70],  # Recovering defender
                [0.30, 0.60],  # Trailing midfielder
                [0.60, 0.45],  # Recovering center-back
                [0.55, 0.55]   # Off-balance recovering defender
            ]),
            'coequipiers': np.array([
                [0.20, 0.40],  # Defensive support
                [0.50, 0.30],  # Right winger on the run
                [0.55, 0.70],  # Left winger on the run 
                [0.65, 0.50],  # Forward making a deep run
                [0.45, 0.50]   # Attacking support
            ]),
            'title': "Fast Counter-Attack (5v5)"
        },
        'surface_reparation': {
            'passeur': (0.85, 0.20),  # Winger ready to cross
            'adversaires': np.array([
                [0.90, 0.25],  # Nearby defender
                [0.87, 0.45],  # Center-back
                [0.92, 0.40],  # Box defender
                [0.80, 0.35],  # Recovering midfielder
                [0.85, 0.60]   # Opposite defender
            ]),
            'coequipiers': np.array([
                [0.70, 0.15],  # Wide support
                [0.80, 0.50],  # Forward at the far post
                [0.88, 0.35],  # Forward at the near post
                [0.75, 0.40],  # Withdrawn attacking midfielder
                [0.65, 0.45]   # Supporting midfielder
            ]),
            'title': "In the Opponent's Box (5v5)"
        },
        'relance_basse': {
            'passeur': (0.08, 0.50),  # Goalkeeper/center-back
            'adversaires': np.array([
                [0.20, 0.30],  # First defensive line
                [0.20, 0.70],  # First defensive line
                [0.35, 0.40],  # Second line
                [0.35, 0.60],  # Second line
                [0.25, 0.50]   # Pressing forward
            ]),
            'coequipiers': np.array([
                [0.15, 0.30],  # Nearby center-back
                [0.15, 0.70],  # Opposite center-back
                [0.25, 0.15],  # Low fullback
                [0.25, 0.85],  # High fullback
                [0.30, 0.50]   # Supporting defensive midfielder
            ]),
            'title': "Build-up from the Back (5v5)"
        },
        'centre_cote': {
            'passeur': (0.95, 0.15),  # Winger in a crossing position
            'adversaires': np.array([
                [0.90, 0.20],  # Defender in direct cover
                [0.85, 0.30],  # Nearby center-back
                [0.85, 0.50],  # Central center-back
                [0.88, 0.70],  # Opposite fullback
                [0.75, 0.40]   # Defensive midfielder
            ]),
            'coequipiers': np.array([
                [0.75, 0.20],  # Rear support
                [0.85, 0.40],  # Near-post forward
                [0.80, 0.60],  # Far-post forward
                [0.70, 0.50],  # Midfielder at the edge of the box
                [0.90, 0.80]   # Opposite fullback
            ]),
            'title': "Cross from the Wing (5v5)"
        }
    }
    return situations.get(situation_name, situations['milieu_terrain'])

def create_pass_heatmap(origin_x, origin_y, adv_positions, teammate_positions, predictor, 
                       x_divisions=15, y_divisions=10):
    """
    Updated version using the functions from Pass_chances_function.py
    """
    # Create a grid of target positions
    x_grid = np.linspace(0, 1, 30)
    y_grid = np.linspace(0, 1, 20)
    
    # Initialize the probability matrix
    prob_matrix = np.zeros((len(y_grid), len(x_grid)))
    
    # For each possible target position
    for i, target_y in enumerate(y_grid):
        for j, target_x in enumerate(x_grid):
            # Build the lists of the two teams to keep
            team_pos = np.array([pos for pos in teammate_positions if distance(pos[0], pos[1], origin_x, origin_y) > 0.005])
            adv_pos = np.array([pos for pos in adv_positions if distance(pos[0], pos[1], origin_x, origin_y) > 0.005])

            # Compute all features with the existing functions
            features = calculate_pass_features(
                origin_x, origin_y, target_x, target_y,
                adv_pos, team_pos,
                x_divisions, y_divisions
            )
            
            # Create a DataFrame with the features
            feature_df = pd.DataFrame([features])
            
            # Predict the probability
            prob = predictor.predict_proba(feature_df)[0, 1]  # Positive class probability
            prob_matrix[i, j] = prob
    
    return x_grid, y_grid, prob_matrix

def configuration_jeu(idx_frame, csv_home, csv_away):
    df_home = pd.read_csv(csv_home, header=2)
    df_away = pd.read_csv(csv_away, header=2)

    # Filter the row corresponding to the desired frame
    home_row = df_home[df_home['Frame'] == idx_frame]
    away_row = df_away[df_away['Frame'] == idx_frame]

    home_pos = []
    away_pos = []

    # For each home player
    if not home_row.empty:
        for col in df_home.columns:
            if col.startswith('Player'):
                col_idx = df_home.columns.get_loc(col)
                # The Y column comes right after
                if col_idx + 1 < len(df_home.columns):
                    y_col = df_home.columns[col_idx + 1]
                    x = home_row.iloc[0][col]
                    y = home_row.iloc[0][y_col]
                    if pd.notna(x) and pd.notna(y):
                        home_pos.append((float(x), float(y)))

    # For each away player
    if not away_row.empty:
        for col in df_away.columns:
            if col.startswith('Player'):
                col_idx = df_away.columns.get_loc(col)
                if col_idx + 1 < len(df_away.columns):
                    y_col = df_away.columns[col_idx + 1]
                    x = away_row.iloc[0][col]
                    y = away_row.iloc[0][y_col]
                    if pd.notna(x) and pd.notna(y):
                        away_pos.append((float(x), float(y)))

    return home_pos, away_pos

def flip_positions(positions: list, pitch_length: float = 1.0, pitch_width: float = 1.0) -> list:
    """
    Flips the players' positions on the pitch along x and y.
    positions : list of (x, y) tuples
    pitch_length : pitch length (default 1.0)
    pitch_width  : pitch width (default 1.0)
    Returns : list of (x_flipped, y_flipped) tuples
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]

def plot_situation_heatmap_metrica(attacker_pos, defender_pos, ball_pos, frame_id,  predictor, figsize=(12, 8), output_dir=str(HEATMAPS_DIR / "metrica")):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(figsize=figsize)
    create_pitch(ax)
    
    # Extract the data
    origin_x, origin_y = ball_pos
    adv_positions = np.array(defender_pos)
    teammate_positions = np.array(attacker_pos)
    
    # Display the players
    plot_player(ax, origin_x, origin_y, 'home', '10')  # Passer with number
    
    for i, (x, y) in enumerate(adv_positions):
        plot_player(ax, x, y, 'away')
    
    for i, (x, y) in enumerate(teammate_positions):
        plot_player(ax, x, y, 'home')
    
    # Create and display the heatmap
    x_grid, y_grid, prob_matrix = create_pass_heatmap(
        origin_x, origin_y, adv_positions, teammate_positions, predictor
    )
    
    X, Y = np.meshgrid(x_grid, y_grid)
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=100)
    contour = ax.contourf(X, Y, prob_matrix, levels=50, cmap=cmap, alpha=0.6)
    
    # Add contours for clarity
    ax.contour(X, Y, prob_matrix, levels=[0.2, 0.5, 0.8], colors='black', linewidths=0.5, alpha=0.7)
    
    # Legend
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Success Probability', rotation=270, labelpad=20)
    
    save_path = os.path.join(output_dir, f'heatmap_{frame_id}.png')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig, ax

def visualize_game_situation(ax, home_pos, away_pos, ball_pos, goal_threat_value, title=None, show_players=True):
    """
    Visualizes a game situation with the players' positions and the goal threat value.
    
    Parameters:
        ax: Matplotlib axes to draw on
        home_pos: List of (x, y) tuples for the home team positions
        away_pos: List of (x, y) tuples for the away team positions
        ball_pos: Tuple (x, y) for the ball position
        goal_threat_value: Goal threat value for this situation
        title: Optional title for the plot
        show_players: Boolean indicating whether the players should be shown
    """
    # Create the pitch
    create_pitch(ax)
    
    # Show the players if requested
    if show_players:
        # Home team (blue dots)
        x_home = [pos[0] for pos in home_pos]
        y_home = [pos[1] for pos in home_pos]
        ax.scatter(x_home, y_home, color='blue', s=100, zorder=2, label='Home Team', alpha=0.7)
        
        # Away team (red dots)
        x_away = [pos[0] for pos in away_pos]
        y_away = [pos[1] for pos in away_pos]
        ax.scatter(x_away, y_away, color='red', s=100, zorder=2, label='Away Team', alpha=0.7)
    
    # Draw the ball (bigger yellow dot)
    if ball_pos is not None and all(p is not None for p in ball_pos):
        ax.scatter(ball_pos[0], ball_pos[1], color='yellow', s=200, zorder=3, edgecolor='black', label='Ball')
    
    # Add the goal threat value
    if goal_threat_value is not None:
        threat_text = f"Goal Threat: {goal_threat_value:.4f}"
        ax.text(0.05, 0.95, threat_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
    
    # Add a title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add a legend
    ax.legend(loc='lower right', fontsize=10)
    
    return ax


def plot_situation_heatmap(situation_name, predictor, figsize=(12, 8)):
    """
    Display a heatmap for a given situation
    """
    # Get the situation
    situation = create_situation(situation_name)
    
    # Configuration
    fig, ax = plt.subplots(figsize=figsize)
    create_pitch(ax)
    
    # Extract the data
    origin_x, origin_y = situation['passeur']
    adv_positions = situation['adversaires']
    teammate_positions = situation['coequipiers']
    
    # Display the players
    plot_player(ax, origin_x, origin_y, 'home', '10')  # Passer with number
    
    for i, (x, y) in enumerate(adv_positions):
        plot_player(ax, x, y, 'away')
    
    for i, (x, y) in enumerate(teammate_positions):
        plot_player(ax, x, y, 'home')
    
    # Create and display the heatmap
    x_grid, y_grid, prob_matrix = create_pass_heatmap(
        origin_x, origin_y, adv_positions, teammate_positions, predictor
    )
    
    X, Y = np.meshgrid(x_grid, y_grid)
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=100)
    contour = ax.contourf(X, Y, prob_matrix, levels=50, cmap=cmap, alpha=0.6)
    
    # Add contours for clarity
    ax.contour(X, Y, prob_matrix, levels=[0.2, 0.5, 0.8], colors='black', linewidths=0.5, alpha=0.7)
    
    # Legend
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Success Probability', rotation=270, labelpad=20)

    plt.tight_layout()
    fig.savefig(str(HEATMAPS_DIR / f'heatmap_{situation_name}.png'), dpi=300)
    return fig, ax

def analyze_pass_features_for_situation(situation_name):
    """
    Analyze the features for a given situation
    """
    situation = create_situation(situation_name)
    origin_x, origin_y = situation['passeur']
    adv_positions = situation['adversaires']
    teammate_positions = situation['coequipiers']
    
    print(f"\n=== Analyse des features pour: {situation['title']} ===")
    print(f"Position du passeur: ({origin_x:.2f}, {origin_y:.2f})")
    print(f"Nombre d'adversaires: {len(adv_positions)}")
    print(f"Nombre de coéquipiers: {len(teammate_positions)}")
    
    # Analyze a few target positions
    test_targets = [
        (0.7, 0.3, "Avant-droit"),
        (0.8, 0.5, "Centre-avant"),
        (0.7, 0.7, "Avant-gauche")
    ]
    
    for target_x, target_y, label in test_targets:
        features = calculate_pass_features(
            origin_x, origin_y, target_x, target_y,
            adv_positions, teammate_positions
        )
        
        print(f"\n--- Cible: {label} ({target_x:.2f}, {target_y:.2f}) ---")
        print(f"Adv. proches départ: {features['nb_adv_proches_depart']}")
        print(f"Adv. sur trajectoire: {features['nb_adv_trajectoire']}")
        print(f"Adv. proches arrivée: {features['nb_adv_proches_arrivee']}")
        print(f"Coéq. proches arrivée: {features['nb_coequipiers_proches_arrivee']}")

# Utility function to test a specific pass
def test_single_pass(x_passeur, y_passeur, x_cible, y_cible, 
                    adv_positions, teammate_positions, predictor):
    """
    Test a specific pass and return the probability with the details
    """
    features = calculate_pass_features(
        x_passeur, y_passeur, x_cible, y_cible,
        adv_positions, teammate_positions
    )
    
    feature_df = pd.DataFrame([features])
    prob = predictor.predict_proba(feature_df)[0, 1]
    
    return prob, features

# Safe loading of the predictor
def load_predictor_safely():
    """Load the predictor with a fallback"""
    try:
        from pass_predictor import PassPredictor
        predictor = PassPredictor().load_model()
        print("✅ PassPredictor chargé avec succès")
        return predictor, "PassPredictor"
    except Exception as e:
        print(f"⚠️ Erreur PassPredictor: {e}")
        print("💡 Utilisation d'un prédicteur simple")
        

# Load the predictor
predictor, predictor_type = load_predictor_safely()
print(f"🎯 Prédicteur utilisé: {predictor_type}")

# Test the hyperparameters
print("🔍 VÉRIFICATION DES HYPERPARAMÈTRES")
print_hyperparameters_info()

# Test a simple prediction
print("\n🧪 Test d'une prédiction:")
try:
    situation = create_situation('aile_droite')
    origin_x, origin_y = situation['passeur']
    adv_positions = situation['adversaires']
    teammate_positions = situation['coequipiers']

    features = calculate_pass_features(
        origin_x, origin_y, 0.8, 0.4,
        adv_positions, teammate_positions
    )
    
    print("✅ Features calculées:")
    for key, value in features.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Prediction test
    feature_df = pd.DataFrame([features])
    prob = predictor.predict_pass_success(feature_df)[0]
    # Display and save all heatmaps for each situation
    situations = [
        'aile_droite', 'milieu_terrain', 'contre_attaque',
        'surface_reparation', 'relance_basse', 'centre_cote'
    ]

    for situation_name in situations:
        print(f"\n=== Génération de la heatmap pour: {situation_name} ===")
        try:
            plot_situation_heatmap(situation_name, predictor)
            analyze_pass_features_for_situation(situation_name)
        except Exception as e:
            print(f"❌ Erreur pour {situation_name}: {e}")

except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
    for frame in df_filtered.itertuples():
        print(f"Frame: {frame.Index}, Type: {frame.Type}, Team: {frame.Team}")
        print(frame._fields)

        ball_pos = (frame._11, frame._12)  # Assuming Start_X and Start_Y are at index 11 and 12
        if frame.Team == 'Home':
            home_pos, away_pos = configuration_jeu(frame._5, csv_home, csv_away)
            if frame.Period == 1:
                home_pos = flip_positions(home_pos)
                away_pos = flip_positions(away_pos)
                ball_pos = flip_positions([ball_pos])[0]
            print(f"Ball Position: {ball_pos}")
            plot_situation_heatmap_metrica(home_pos, away_pos, ball_pos, frame.Index, predictor)


        if frame.Team == 'Away':
            home_pos, away_pos = configuration_jeu(frame._5, csv_home, csv_away)
            if frame.Period == 2:
                home_pos = flip_positions(home_pos)
                away_pos = flip_positions(away_pos)
                ball_pos = flip_positions([ball_pos])[0]
            print(f"Ball Position: {ball_pos}")
            plot_situation_heatmap_metrica(away_pos, home_pos, ball_pos, frame.Index, predictor)
