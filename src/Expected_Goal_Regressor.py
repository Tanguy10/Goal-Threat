import numpy as np
import pickle
from xgboost import XGBClassifier
import pandas as pd
from config import MODELS_DIR

def nb_adv_trajectoire_coords(x, y, adv_positions, goal_x=1, goal_y=0.5, seuil_trajectoire=0.016622456647036446):
    """Improved version that accepts coordinates directly"""
    passeur = np.array([x, y])
    cible = np.array([goal_x, goal_y])

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

def defensive_pressure(x, y, adv_positions, sigma=0.020264979108142053):
    """
    Calculate defensive pressure as weighted sum of proximity to defenders
    
    Args:
        x, y: Player position
        adv_positions: Numpy array of shape (n, 2) containing opponents positions
        sigma: Parameter for Gaussian weighting (smaller = steeper drop-off)
        
    Returns:
        Defensive pressure value
    """
    if len(adv_positions) == 0:
        return 0
    
    player_pos = np.array([x, y])
    distances = np.linalg.norm(adv_positions - player_pos, axis=1)
    
    # Use Gaussian weighting: closer defenders have more influence
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    
    return np.sum(weights)

class GoAloneXGBPredictor:
    def __init__(self, model_path=None, metadata_path=None):
        model_path = str(model_path or MODELS_DIR / "best_xg_model_xgb.json")
        metadata_path = str(metadata_path or MODELS_DIR / "xg_model_metadata.pkl")
        # Load the XGBoost model
        self.model = XGBClassifier()  # or XGBRegressor depending on the model
        self.model.load_model(model_path)
        # Load the scaler and the features
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        self.scaler = metadata["scaler"]
        self.features = metadata["features"]

    def go_alone(self, ball_pos, defenders_pos):
        x, y = ball_pos
        adv_positions = np.array(defenders_pos) if defenders_pos else np.zeros((0,2))
        angle_tir = self._angle_to_goal(x, y)
        dist_but = self._distance_to_goal(x, y)
        nb_defenders = nb_adv_trajectoire_coords(x, y, adv_positions)
        def_pressure = defensive_pressure(x, y, adv_positions)
        X = pd.DataFrame([[angle_tir, dist_but, nb_defenders, def_pressure]], columns=self.features)
        X_scaled = self.scaler.transform(X)
        xg_pred = self.model.predict_proba(X_scaled)[0,1] if hasattr(self.model, "predict_proba") else self.model.predict(X_scaled)[0]
        return float(xg_pred)

    def _distance_to_goal(self, x, y, goal_x=1, goal_y=0.5):
        return np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

    def _angle_to_goal(self, x, y, goal_x=1, goal_y1=0.44, goal_y2=0.56):
        a = np.array([goal_x, goal_y1])
        b = np.array([goal_x, goal_y2])
        p = np.array([x, y])
        pa = a - p
        pb = b - p
        norm_pa = np.linalg.norm(pa)
        norm_pb = np.linalg.norm(pb)
        if norm_pa == 0 or norm_pb == 0:
            return np.pi
        cos_angle = np.dot(pa, pb) / (norm_pa * norm_pb)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        return angle

# Example usage
if __name__ == "__main__":
    predictor = GoAloneXGBPredictor()
    ball_pos = (0.9, 0.5)
    defenders_pos = [(0.8, 0.5), (0.95, 0.52)]
    print(predictor.go_alone(ball_pos, defenders_pos))
