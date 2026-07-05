import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

from Go_alone_simple import GoAloneSimplePredictor
from config import MODELS_DIR, PROCESSED_DIR, FIGURES_DIR

def load_shot_data(csv_path):
    """
    Load and keep only the shot rows from the CSV
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Keep only the rows containing shots
    # Adjust this filter depending on your CSV structure
    shots_df = df[df['tir_observe'] == 1].copy()
    
    print(f"Total d'entrées: {len(df)}, Nombre de tirs: {len(shots_df)}")
    return shots_df

def extract_features_from_shot_df(shot_df):
    """
    Extract the information needed for the prediction
    """
    features = []
    
    for idx, row in shot_df.iterrows():
        # Skip rows without xG_tir
        if pd.isna(row['xG_tir']):
            continue

        player_x = row['x']
        player_y = row['y']
        defenders_coords = []
        xg = row['xG_tir']
        is_direct_shot = False
        
        features.append({
            'player_x': player_x,
            'player_y': player_y,
            'defenders_coords': defenders_coords,
            'goal_x': 1.0,
            'goal_y': 0.5,
            'is_direct_shot': is_direct_shot,
            'actual_xg': xg
        })
    
    return features

def objective_function(params, features):
    """
    Function to minimize: error between predictions and actual xG
    """
    # Unpack all parameters
    sigma_close, sigma_trajectory, sigma_angle, \
    distance_weight, angle_weight, close_density_weight, trajectory_density_weight, \
    direct_shot_bonus = params
    
    # Check that the sum of the weights is 1
    weights_sum = distance_weight + angle_weight + close_density_weight + trajectory_density_weight
    if not np.isclose(weights_sum, 1.0, atol=0.01):
        # Large penalty if the weights do not sum to 1
        return 1000.0
    
    predictor = GoAloneSimplePredictor(
        normalized_coords=True,
        sigma_close=sigma_close,
        sigma_trajectory=sigma_trajectory,
        sigma_angle=sigma_angle,
        distance_weight=distance_weight,
        angle_weight=angle_weight,
        close_density_weight=close_density_weight,
        trajectory_density_weight=trajectory_density_weight,
        direct_shot_bonus=direct_shot_bonus
    )
    
    actual_xgs = []
    predicted_xgs = []
    
    for feature in features:
        result = predictor.predict_success_probability(
            feature['player_x'], 
            feature['player_y'],
            feature['defenders_coords'],
            goal_x=feature['goal_x'],
            goal_y=feature['goal_y'],
            is_direct_shot=feature['is_direct_shot']
        )
        predicted_xg = result['probability']
        
        actual_xgs.append(feature['actual_xg'])
        predicted_xgs.append(predicted_xg)
    
    # Compute the mean squared error
    mse = mean_squared_error(actual_xgs, predicted_xgs)
    return mse

def optimize_parameters(features, initial_params=None):
    """
    Optimize all parameters to minimize the error
    """
    if initial_params is None:
        # Default initial values
        initial_params = (
            0.05,     # sigma_close - different
            0.12,     # sigma_trajectory - different
            np.pi/3,  # sigma_angle - different
            0.30,     # etc...
            0.25,
            0.20,
            0.25,
            0.10
        )

    
    # Bounds for the parameters
    bounds = [
        (0.01, 0.5),     # sigma_close: positive, relatively small values
        (0.01, 0.5),     # sigma_trajectory: positive, relatively small values
        (np.pi/12, np.pi/2),  # sigma_angle: between 15 and 90 degrees
        (0.1, 0.6),      # distance_weight: reasonable weight
        (0.1, 0.6),      # angle_weight: reasonable weight
        (0.1, 0.6),      # close_density_weight: reasonable weight
        (0.1, 0.6),      # trajectory_density_weight: reasonable weight
        (0.0, 0.2)       # direct_shot_bonus: small bonus
    ]
    
    print(f"Début de l'optimisation avec valeurs initiales:")
    print(f"- sigma_close: {initial_params[0]:.4f}")
    print(f"- sigma_trajectory: {initial_params[1]:.4f}")
    print(f"- sigma_angle: {initial_params[2]:.4f} (radians)")
    print(f"- distance_weight: {initial_params[3]:.4f}")
    print(f"- angle_weight: {initial_params[4]:.4f}")
    print(f"- close_density_weight: {initial_params[5]:.4f}")
    print(f"- trajectory_density_weight: {initial_params[6]:.4f}")
    print(f"- direct_shot_bonus: {initial_params[7]:.4f}")
    
    # Use a global optimization algorithm to avoid local minima
    result = minimize(
        objective_function,
        initial_params,
        args=(features,),
        bounds=bounds,
        method='L-BFGS-B',  # Or try 'SLSQP' which handles constraints better
    )
    
    if result.success:
        print(f"Optimisation réussie: {result.message}")
        print(f"Paramètres optimaux trouvés:")
        print(f"- sigma_close = {result.x[0]:.5f}")
        print(f"- sigma_trajectory = {result.x[1]:.5f}")
        print(f"- sigma_angle = {result.x[2]:.5f} radians ({np.degrees(result.x[2]):.2f}°)")
        print(f"- distance_weight = {result.x[3]:.5f}")
        print(f"- angle_weight = {result.x[4]:.5f}")
        print(f"- close_density_weight = {result.x[5]:.5f}")
        print(f"- trajectory_density_weight = {result.x[6]:.5f}")
        print(f"- direct_shot_bonus = {result.x[7]:.5f}")
        print(f"Somme des poids: {result.x[3] + result.x[4] + result.x[5] + result.x[6]:.5f}")
        print(f"Erreur quadratique moyenne: {result.fun:.5f}")
        
        return result.x
    else:
        print(f"L'optimisation a échoué: {result.message}")
        return initial_params

def main():
    # Path to your database
    shot_database_path = str(PROCESSED_DIR / "shot_opportunities_database.csv")
    
    print("Chargement des données de tirs...")
    shots_df = load_shot_data(shot_database_path)
    
    print("Extraction des caractéristiques...")
    features = extract_features_from_shot_df(shots_df)
    
    print(f"Nombre de tirs avec caractéristiques extraites: {len(features)}")
    
    print("Optimisation des paramètres...")
    optimal_params = optimize_parameters(features)
    
    # Create a predictor with the optimized parameters
    optimized_predictor = GoAloneSimplePredictor(
        normalized_coords=True,
        sigma_close=optimal_params[0],
        sigma_trajectory=optimal_params[1],
        sigma_angle=optimal_params[2],
        distance_weight=optimal_params[3],
        angle_weight=optimal_params[4],
        close_density_weight=optimal_params[5],
        trajectory_density_weight=optimal_params[6],
        direct_shot_bonus=optimal_params[7]
    )
    
    # Visualize the results
    actual_xgs = [f['actual_xg'] for f in features]
    predicted_xgs = [optimized_predictor.predict_success_probability(
        f['player_x'], f['player_y'], f['defenders_coords'], 
        f['goal_x'], f['goal_y'], f['is_direct_shot']
    )['probability'] for f in features]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_xgs, predicted_xgs, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('xG réel')
    plt.ylabel('xG prédit')
    plt.title('Comparaison entre xG réel et xG prédit (paramètres optimisés)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / 'optimization_results_full.png'))
    plt.show()
    
    # Save the optimized parameters to a file
    params_dict = {
        'sigma_close': float(optimal_params[0]),
        'sigma_trajectory': float(optimal_params[1]),
        'sigma_angle': float(optimal_params[2]),
        'distance_weight': float(optimal_params[3]),
        'angle_weight': float(optimal_params[4]),
        'close_density_weight': float(optimal_params[5]),
        'trajectory_density_weight': float(optimal_params[6]),
        'direct_shot_bonus': float(optimal_params[7])
    }
    
    import json
    with open(str(MODELS_DIR / 'optimized_parameters.json'), 'w') as f:
        json.dump(params_dict, f, indent=4)
    print("Paramètres optimisés sauvegardés dans 'optimized_parameters.json'")

if __name__ == "__main__":
    main()