import numpy as np
from itertools import product
from Pass_chances_function import get_zone, densite_adversaires_ponderee, nb_adv_trajectoire_coords, diff_distance_joueurs_proches
from pass_predictor import PassPredictor
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from Go_alone_simple import GoAloneSimplePredictor
from Expected_Goal_Regressor import GoAloneXGBPredictor
from config import MODELS_DIR
import os
import joblib
import pickle
from functools import lru_cache
import pandas as pd




def distance(x,y):
    """
    Compute the Euclidean distance between two points (x, y).
    
    Args:
        x (Tuple[float, float]): Coordinates of the first point.
        y (Tuple[float, float]): Coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def is_off_side( player_pos: Tuple[float, float],defenders_pos: List[Tuple[float, float]], ball_pos: Tuple[float, float]) -> bool:
    """
    Check whether a player is in an offside position.

    Args:
        player_pos (Tuple[float, float]): Player coordinates.
        defenders_pos (List[Tuple[float, float]]): Defender coordinates.
        ball_pos (Tuple[float, float]): Ball coordinates.

    Returns:
        bool: True if the player is offside, False otherwise.
    """
    if ball_pos[0] > player_pos[0]:  # The ball is in front of the player
        return False

    # Check whether the player is behind the defenders' line
    number =0
    for defender in defenders_pos:
        if defender[0] < player_pos[0]:  # A defender is behind the player
            number += 1

    return number > 2


class MLShotPredictor:
    """Shot predictor using an ML model with only the 4 base features"""
    
    VALID_MODELS = ['LightGBM', 'XGBoost', 'RandomForest', 'LogisticRegression']
    
    def __init__(self, model_type='LogisticRegression', model_path=None):
        """
        Initialize the predictor by loading the specified model
        
        Args:
            model_type: Type of model to use ('LightGBM', 'XGBoost', 'RandomForest', 'LogisticRegression')
            model_path: Path to the model file (pkl or txt); if None, use the default model
        """
        self.model = None
        self.model_type = model_type if model_type in self.VALID_MODELS else 'LogisticRegression'
        self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """
        Load the model from a file
        
        If model_path is None, look for the specified model among the default files
        """
        if model_path is None:
            # Default path
            models_dir = str(MODELS_DIR)

            # Look for the specific model
            if self.model_type == 'LightGBM':
                model_path = os.path.join(models_dir, "shot_model_lightgbm.txt")
                if not os.path.exists(model_path):
                    model_path = os.path.join(models_dir, "best_shot_model.txt")
            else:
                # For sklearn models (XGBoost, RandomForest, LogisticRegression)
                model_path = os.path.join(models_dir, f"shot_model_{self.model_type.lower()}.pkl")
                if not os.path.exists(model_path):
                    model_path = os.path.join(models_dir, "best_shot_model.pkl")

            # If still not found, use the default model
            if not os.path.exists(model_path):
                model_path = os.path.join(models_dir, "go_alone_model.txt")
                self.model_type = 'LightGBM'  # Fallback to LightGBM
        
        # Load the model
        try:
            if model_path.endswith('.txt'):
                # For LightGBM models (.txt)
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=model_path)
                self.model_type = 'LightGBM'
            else:
                # For sklearn models (.pkl)
                self.model = joblib.load(model_path)
                # Detect the model type
                if hasattr(self.model, '__class__'):
                    model_class = self.model.__class__.__name__
                    if 'XGBoost' in model_class:
                        self.model_type = 'XGBoost'
                    elif 'RandomForest' in model_class:
                        self.model_type = 'RandomForest'
                    elif 'LogisticRegression' in model_class:
                        self.model_type = 'LogisticRegression'
            
            print(f"Shot predictor loaded: {self.model_type} model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        return self
    
    @lru_cache(maxsize=1024)
    def _predict_xg_cached(self, player_x, player_y, defenders_tuple):
        """Cached version of predict_xg for the 4 base features"""
        # Convert the tuple to a list
        defenders_coords = [d for d in defenders_tuple] if defenders_tuple else []
        
        # Compute only the 4 base features
        features = self._calculate_basic_features(player_x, player_y, defenders_coords)
        
        # Predict with the appropriate model
        if self.model_type == 'LightGBM':
            return self.model.predict(features)[0]
        else:
            # XGBoost, RandomForest, LogisticRegression
            return self.model.predict_proba(features)[0, 1]
    
    def predict_xg(self, player_x, player_y, defenders_coords=None):
        """
        Predict the probability of scoring a goal (expected goals) with the 4 base features
        
        Args:
            player_x: Shooter x position (normalized)
            player_y: Shooter y position (normalized)
            defenders_coords: List of defender positions [(x1,y1), (x2,y2), ...]
        
        Returns:
            float: Probability of scoring
        """
        if defenders_coords is None:
            defenders_coords = []
        
        # Convert to a hashable tuple for caching
        defenders_tuple = tuple(tuple(d) for d in defenders_coords)
        
        # Call the cached version
        return self._predict_xg_cached(player_x, player_y, defenders_tuple)
    
    def _calculate_basic_features(self, player_x, player_y, defenders_coords):
        """
        Compute only the 4 base features for the prediction
        
        Returns:
            numpy.array: Feature vector
        """
        # 1. Distance to the goal line
        dist_to_goal_line = 1 - player_x
        
        # 2. Distance to the center
        dist_from_center = abs(player_y - 0.5)
        
        # 3. Defensive density (distance to the closest defender)
        if defenders_coords:
            distances = []
            for d in defenders_coords:
                # Scale to match the original units
                dx = (player_x - d[0]) * 120
                dy = (player_y - d[1]) * 80
                distance = np.sqrt(dx*dx + dy*dy)
                distances.append(distance)
            density = min(distances) if distances else 1.0
        else:
            density = 1.0
            
        # 4. Defenders on the trajectory
        if defenders_coords:
            def_coord = np.array(defenders_coords)
            defenders_in_path = nb_adv_trajectoire_coords(
                player_x, player_y, 1.0, 0.5, def_coord)
        else:
            defenders_in_path = 0
        
        # Create the feature vector with only the 4 base features
        return np.array([[
            dist_to_goal_line, 
            dist_from_center,
            density,
            defenders_in_path
        ]])

predictor = MLShotPredictor(model_type='LogisticRegression')

class Zone:
    """Classe pour représenter une zone du terrain"""
    
    def __init__(self, x: float, y: float):
        self.x = max(0.0, min(1.0, x))  # Limiter entre 0 et 1
        self.y = max(0.0, min(1.0, y))
    
    def __iter__(self):
        """Allows using tuple(zone) or x, y = zone"""
        return iter((self.x, self.y))
    
    def center(self) -> Tuple[float, float]:
        """Return the zone center as a tuple (x, y)"""
        return (self.x, self.y)
    
    def __getitem__(self, index):
        """Allows access via zone[0] and zone[1]"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Zone index out of range")
    
    def __eq__(self, other):
        if isinstance(other, Zone):
            return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
        return False
    
    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))
    
    def __repr__(self):
        return f"Zone({self.x:.3f}, {self.y:.3f})"
    
    def distance_to(self, other: 'Zone') -> float:
        """Compute the Euclidean distance to another zone"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def is_closer_to_goal_than(self, other: 'Zone', goal: 'Zone' = None) -> bool:
        """Check whether this zone is closer to goal than another"""
        if goal is None:
            goal = Zone(1.0, 0.5)
        return self.distance_to(goal) < other.distance_to(goal)
    
    @classmethod
    def from_tuple(cls, pos: Tuple[float, float]) -> 'Zone':
        """Create a zone from a tuple"""
        return cls(pos[0], pos[1])

class Player:
    """Class representing a player"""
    
    def __init__(self, position: Zone, team: str = "unknown", player_id: int = 0):
        self.position = position
        self.team = team  # "teammates" or "defenders"
        self.player_id = player_id
    
    def __repr__(self):
        return f"Player({self.team}_{self.player_id}, {self.position})"
    
    def move_towards(self, target: Zone, max_distance: float) -> 'Player':
        """Create a new player moved toward a target"""
        dx = target.x - self.position.x
        dy = target.y - self.position.y
        current_distance = np.sqrt(dx**2 + dy**2)
        
        if current_distance == 0:
            return Player(self.position, self.team, self.player_id)
        
        movement_ratio = min(max_distance / current_distance, 1.0)
        new_x = self.position.x + dx * movement_ratio
        new_y = self.position.y + dy * movement_ratio
        
        return Player(Zone(new_x, new_y), self.team, self.player_id)

class GameState:
    """Class representing the full game state"""
    
    def __init__(self, ball_zone: Zone, teammates: List[Player], defenders: List[Player]):
        self.ball_zone = ball_zone
        self.teammates = teammates
        self.defenders = defenders
        self.goal = Zone(1.0, 0.5)
    
    def __repr__(self):
        return f"GameState(ball={self.ball_zone}, teammates={len(self.teammates)}, defenders={len(self.defenders)})"
    
    @classmethod
    def from_positions(cls, ball_pos: Tuple[float, float], 
                      teammates_pos: List[Tuple[float, float]], 
                      defenders_pos: List[Tuple[float, float]]) -> 'GameState':
        """Create a GameState from lists of positions"""
        ball_zone = Zone.from_tuple(ball_pos)
        teammates = [Player(Zone.from_tuple(pos), "teammates", i) 
                    for i, pos in enumerate(teammates_pos)]
        defenders = [Player(Zone.from_tuple(pos), "defenders", i) 
                    for i, pos in enumerate(defenders_pos)]
        return cls(ball_zone, teammates, defenders)

class Terrain:
    """Class to manage the pitch and its zones"""

    def __init__(self, x_divisions: int = 15, y_divisions: int = 10):
        self.x_divisions = x_divisions
        self.y_divisions = y_divisions
        self.all_zones = self._create_all_zones()
        self.goal = Zone(1.0, 0.5)
        # Compute the size of each zone
        self.zone_width = 1.0 / self.x_divisions
        self.zone_height = 1.0 / self.y_divisions
    
    def _create_all_zones(self) -> List[Zone]:
        """Create all the pitch zones"""
        zones = []
        zone_width = 1.0 / self.x_divisions
        zone_height = 1.0 / self.y_divisions
        
        for i in range(self.x_divisions):
            for j in range(self.y_divisions):
                center_x = (i + 0.5) * zone_width
                center_y = (j + 0.5) * zone_height
                zones.append(Zone(center_x, center_y))
        
        return zones
    
    def get_reachable_zones(self, from_zone: Zone) -> List[Zone]:
        """Return all zones reachable from a given zone"""
        if from_zone.center()[0]< 0.85:
            return [zone for zone in self.all_zones if zone.center()[0] > from_zone.center()[0] +0.05]
        else:
            l = truth_zone.copy()
            l.extend(zone for zone in self.all_zones if abs(zone.center()[1] -0.5) < abs(from_zone.center()[1] -0.5) and zone.center()[0] > from_zone.center()[0])
            return l

    def sample_points_in_zone(self, zone: Zone, num_samples: int = 4) -> List[Tuple[float, float]]:
        """
        Sample several random points in a given zone
        
        Args:
            zone: Zone to sample (its center is already known)
            num_samples: Number of points to sample
            
        Returns:
            List of sampled points (Zone)
        """
        # Known cell dimensions
        half_width = 1.0 / (2 * self.x_divisions)
        half_height = 1.0 / (2 * self.y_divisions)
        
        # Center coordinates
        x_center = zone.x
        y_center = zone.y
        
        # Zone bounds
        min_x = max(0.0, x_center - half_width)
        max_x = min(1.0, x_center + half_width)
        min_y = max(0.0, y_center - half_height)
        max_y = min(1.0, y_center + half_height)
        
        # List of points to return
        points = []
        
        # Always include the center
        points.append((x_center, y_center))
        
        # Generate random points to fill up to num_samples
        while len(points) < num_samples:
            x = min_x + np.random.random() * (max_x - min_x)
            y = min_y + np.random.random() * (max_y - min_y)
            points.append((x, y))
        
        return points

def truth_zone():
    """
    Return the list of zones from which a direct shot is the optimal choice.
    Now forms a cone that narrows as it approaches the goal.
    
    Returns:
        List[Zone]: Zones where the direct shot is optimal
    """
    zones_optimal_shot = []
    terrain = Terrain()
    for zone in terrain.all_zones:
        x, y = zone.center()
        # The closer to goal, the narrower the zone
        if x > 0.85:
            # Compute the allowed width as a function of x
            # The closer x is to 1.0 (goal), the smaller the width
            max_width = 0.2 * (1.0 - x) / 0.1 + 0.05
            
            # Check whether the zone is inside the cone
            if abs(y - 0.5) < max_width:
                zones_optimal_shot.append(zone)
    return zones_optimal_shot

truth_zone = truth_zone()
print(truth_zone)

class MovementCalculator:
    """Class to compute player movements"""
    
    def __init__(self, player_speed: float = 5.0, ball_speed: float = 15.0):
        self.player_speed = player_speed
        self.ball_speed = ball_speed
    
    def calculate_pass_time(self, from_zone: Zone, to_zone: Zone) -> float:
        """Compute the time needed for a pass"""
        pass_distance = from_zone.distance_to(to_zone)
        pass_distance_meters = pass_distance * 120  # 120m pitch
        return pass_distance_meters / self.ball_speed
    
    def calculate_max_movement(self, from_zone: Zone, to_zone: Zone) -> float:
        """Compute the maximum distance a player can cover during a pass"""
        pass_time = self.calculate_pass_time(from_zone, to_zone)
        max_movement_meters = self.player_speed * pass_time
        return min(max_movement_meters / 120, 0.1)  # Normalize and cap
    
    def generate_circle_positions(self, center: Zone, radius: float, num_points: int = 10) -> List[Zone]:
        """Generate positions within a circle around a center"""
        positions = [center]
        
        if radius > 0:
            for _ in range(num_points):
                angle = np.random.uniform(0, 2*np.pi)
                r = radius * np.sqrt(np.random.uniform(0, 1))
                
                new_x = center.x + r * np.cos(angle)
                new_y = center.y + r * np.sin(angle)
                
                positions.append(Zone(new_x, new_y))
        
        return positions

class ImportantPlayersIdentifier:
    """Class to identify the important players in an action"""
    
    def __init__(self, distance_threshold: float = 0.15, forward_threshold: float = 0.05):
        self.distance_threshold = distance_threshold
        self.forward_threshold = forward_threshold
    
    def identify(self, game_state: GameState, pass_target: Zone) -> Tuple[List[int], List[int]]:
        """Identify the important players for a given pass"""
        action_center = Zone(
            (game_state.ball_zone.x + pass_target.x) / 2,
            (game_state.ball_zone.y + pass_target.y) / 2
        )
        max_x_action = max(game_state.ball_zone.x, pass_target.x)
        
        important_defenders = []
        for i, player in enumerate(game_state.defenders):
            if self._is_important_player(player, game_state.ball_zone, pass_target, 
                                       action_center, max_x_action):
                important_defenders.append(i)
        
        important_teammates = []
        for i, player in enumerate(game_state.teammates):
            if self._is_important_player(player, game_state.ball_zone, pass_target, 
                                       action_center, max_x_action):
                important_teammates.append(i)
        
        return important_defenders, important_teammates
    
    def _is_important_player(self, player: Player, ball_zone: Zone, pass_target: Zone,
                           action_center: Zone, max_x_action: float) -> bool:
        """Check whether a player is important for the action"""
        # Criterion 1: Proximity to the action
        distances = [
            player.position.distance_to(ball_zone),
            player.position.distance_to(pass_target),
            player.position.distance_to(action_center)
        ]
        
        if min(distances) < self.distance_threshold:
            return True
        
        # Criterion 2: Player further up the pitch
        if player.position.x > max_x_action + self.forward_threshold:
            return True
        
        return False

class ConfigurationGenerator:
    """Class to generate player configurations"""
    
    def __init__(self, movement_calculator: MovementCalculator, max_configs: int = 100):
        self.movement_calculator = movement_calculator
        self.max_configs = max_configs

    def generate_configurations(self, game_state: GameState, pass_target: Zone,
                            defenders: List[int], teammates: List[int],
                            num_samples: int = 3) -> List[GameState]:
        """Generate all possible player configurations"""
        max_movement_def = 0.02
        max_movement_att = 0.03
        goal = Zone(1.0, 0.5)
        
        # 1. Identify the attacker closest to pass_target
        closest_attacker_idx = -1
        min_distance = float('inf')
        for i, teammate in enumerate(game_state.teammates):
            dist = teammate.position.distance_to(pass_target)
            if dist < min_distance:
                min_distance = dist
                closest_attacker_idx = i
        
        # 2. Identify the 3 attackers closest to the goal
        advanced_attackers = sorted(
            [(i, player) for i, player in enumerate(game_state.teammates) if i != closest_attacker_idx],
            key=lambda x: x[1].position.distance_to(game_state.goal),
        )[:3]
        advanced_attackers_idx = [idx for idx, _ in advanced_attackers]
        
        # 3. Prepare the new positions for each type of player
        new_teammates = []
        for i, teammate in enumerate(game_state.teammates):
            if i == closest_attacker_idx:
                # The closest attacker goes exactly to the pass position
                new_teammates.append(Player(pass_target, "teammates", teammate.player_id))
            elif i in advanced_attackers_idx and teammate.position.x < 0.96:
                # The 3 most advanced attackers move toward the goal with a bit of randomness
                target_pos = teammate.move_towards(goal, max_movement_att).position
                # Generate a few positions around the target point
                possible_positions = self.movement_calculator.generate_circle_positions(target_pos, radius=0.02, num_points=num_samples)
                # Choose a random position among these possibilities
                chosen_pos = random.choice(possible_positions)
                new_teammates.append(Player(chosen_pos, "teammates", teammate.player_id))
            else:
                # The other attackers stay in the same place
                new_teammates.append(teammate)
        
        # 4. For the defenders, recenter on the ball-goal axis
        new_defenders = []
        # Compute the 1/3 point between the pass target and the goal
        target_x = (2/3) * pass_target.x + (1/3) * goal.x
        target_y = (2/3) * pass_target.y + (1/3) * goal.y
        axis_point = Zone(target_x, target_y)
        for defender in game_state.defenders:
            # Move the defender toward this point
            if defender.position.x < 1.0:
                moved_defender = defender.move_towards(axis_point, max_movement_def)
                new_defenders.append(moved_defender)
            else :
                new_defenders.append(defender)

        # Create a single configuration with these new positions
        return [GameState(pass_target, new_teammates, new_defenders)]

    def _generate_player_variations(self, players: List[Player], important_indices: List[int],
                                  max_movement: float, target: Zone, num_samples: int,
                                  behavior: str) -> List[List[Zone]]:
        """Generate position variations for a list of players"""
        variations = []
        
        for i, player in enumerate(players):
            if i in important_indices:
                # Important player: random positions within a circle
                positions = self.movement_calculator.generate_circle_positions(
                    player.position, max_movement, num_samples
                )
                variations.append(positions)
            else:
                # Non-important player: predefined movement
                if behavior == "static":
                    new_pos = player.position
                else:
                    new_player = player.move_towards(target, max_movement)
                    new_pos = new_player.position
                variations.append([new_pos])
        
        return variations

class PassPredictor:
    """Interface for pass prediction"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        from pass_predictor import PassPredictor as PP
        self.predictor = PP().load_model()
    
    def predict_probability(self, from_point, to_point, game_state: GameState) -> float:
        """
        Predict the success probability of a pass between two points
        
        Args:
            from_point: Start point (can be Zone or tuple)
            to_point: End point (can be Zone or tuple)
            game_state: Game state
            
        Returns:
            float: Pass success probability
        """
        # Extract the x, y coordinates - supports both Zone objects and tuples
        from_x = from_point.x if hasattr(from_point, 'x') else from_point[0]
        from_y = from_point.y if hasattr(from_point, 'y') else from_point[1]
        to_x = to_point.x if hasattr(to_point, 'x') else to_point[0]
        to_y = to_point.y if hasattr(to_point, 'y') else to_point[1]
        
        # Convert the players' positions
        defenders_pos = [(p.position.x, p.position.y) for p in game_state.defenders]
        
        # Only consider teammates who are not offside
        teammates_pos = []
        for teammate in game_state.teammates:
            if not is_off_side(
                (teammate.position.x, teammate.position.y),
                defenders_pos,
                (from_x, from_y)
            ):
                teammates_pos.append((teammate.position.x, teammate.position.y))
        
        r_config = [defenders_pos, teammates_pos]
        
        # Compute the features - create a temporary Zone object if needed
        if hasattr(from_point, 'x') and hasattr(to_point, 'x'):
            features = self._calculate_features(from_point, to_point, r_config)
        else:
            from_zone = Zone(from_x, from_y)
            to_zone = Zone(to_x, to_y)
            features = self._calculate_features(from_zone, to_zone, r_config)
        
        # Create a DataFrame with the direct coordinates
        pass_data = pd.DataFrame({
            'x_passeur': [from_x],
            'y_passeur': [from_y],
            'x_cible': [to_x],
            'y_cible': [to_y],
            'nb_adv_proches_depart': [features['nb_adv_proches_depart']],
            'nb_adv_trajectoire': [features['nb_adv_trajectoire']],
            'nb_adv_proches_arrivee': [features['nb_adv_proches_arrivee']],
            'nb_coequipiers_proches_arrivee': [features['nb_coequipiers_proches_arrivee']],
            'diff_distance_normalisee': [features['diff_distance_normalisee']]
        })
        
        return self.predictor.predict_pass_success(pass_data)[0]
    
    def _calculate_features(self, from_point, to_point, r_config) -> dict:
        """
        Compute the features for the prediction, works with arbitrary points
        
        Args:
            from_point: Start point (can be Zone or tuple)
            to_point: End point (can be Zone or tuple)
            r_config: Configuration [defenders_positions, teammates_positions]
            
        Returns:
            dict: Features computed for the pass
        """
        # Extract the coordinates - supports both Zone objects and tuples
        from_x = from_point.x if hasattr(from_point, 'x') else from_point[0]
        from_y = from_point.y if hasattr(from_point, 'y') else from_point[1]
        to_x = to_point.x if hasattr(to_point, 'x') else to_point[0]
        to_y = to_point.y if hasattr(to_point, 'y') else to_point[1]
        
        defenders_positions, teammates_positions = r_config

        defenders_array = np.array([pos for pos in defenders_positions if distance((from_x, from_y), pos) > 0.005])
        teammates_array = np.array([pos for pos in teammates_positions if distance((from_x, from_y), pos) > 0.005])

        optimal_params = self.predictor.get_optimal_params()
        sigma = optimal_params.get('sigma', 0.05) if optimal_params else 0.05
        seuil_trajectoire = optimal_params.get('seuil_trajectoire', 0.02) if optimal_params else 0.02
        
        # Distance computation between the points
        distance_passe = np.sqrt((from_x - to_x)**2 + (from_y - to_y)**2)
        
        features = {}
        features['distance_passe'] = distance_passe
        features['sens_passe'] = 1 if to_x > from_x else -1
        
        if len(defenders_array) > 0:
            features['nb_adv_proches_depart'] = densite_adversaires_ponderee(
                from_x, from_y, defenders_array, sigma
            )
            features['nb_adv_trajectoire'] = nb_adv_trajectoire_coords(
                from_x, from_y, to_x, to_y,
                defenders_array, seuil_trajectoire
            )
            features['nb_adv_proches_arrivee'] = densite_adversaires_ponderee(
                to_x, to_y, defenders_array, sigma
            )
            features['diff_distance_normalisee'] = diff_distance_joueurs_proches(
                to_x, to_y, defenders_array, teammates_array, from_x, from_y
            )
        else:
            features['nb_adv_proches_depart'] = 0.0
            features['nb_adv_trajectoire'] = 0
            features['nb_adv_proches_arrivee'] = 0.0
        
        if len(teammates_array) > 0:
            features['nb_coequipiers_proches_arrivee'] = densite_adversaires_ponderee(
                to_x, to_y, teammates_array, sigma
            )
        else:
            features['nb_coequipiers_proches_arrivee'] = 0.0
        
        return features
     
class ShotPredictor:
    """Class to predict goal chances using the GoAloneSimplePredictor model"""
    
    @staticmethod
    # def predict_probability(ball_zone: Zone, goal: Zone = None, defenders: List[Player] = None) -> float:
    #     """
    #     Predict the probability of scoring from a zone using the GoAloneSimplePredictor
    #     
    #     Args:
    #         ball_zone: Zone where the ball is
    #         goal: Goal zone (default 1.0, 0.5)
    #         defenders: List of defenders to consider
    #         
    #     Returns:
    #         float: Probability of scoring a goal
    #     """
    #     if goal is None:
    #         goal = Zone(1.0, 0.5)
    #         
    #     # If no defender is provided, use the simplified distance-based approach
    #     if defenders is None or len(defenders) == 0:
    #         distance_to_goal = ball_zone.distance_to(goal)
    #         
    #         if distance_to_goal < 0.1:
    #             return 1.0
    #         elif distance_to_goal < 0.2:
    #             return 0.7
    #         else:
    #             return 0.3
    #     
    #     # Convert the defender positions to the format expected by GoAloneSimplePredictor
    #     defenders_coords = [(d.position.x, d.position.y) for d in defenders]
    #     
    #     # Use the GoAloneSimplePredictor
    #     result = predictor.predict_success_probability(
    #         player_x=ball_zone.x,
    #         player_y=ball_zone.y,
    #         defenders_coords=defenders_coords,
    #         goal_x=goal.x,
    #         goal_y=goal.y
    #     )
        
        # Return the computed probability
        # return result['probability']

    def predict_probability(ball_zone: Zone, goal: Zone = None, defenders: List[Player] = None) -> float:
        if goal is None:
            goal = Zone(1.0, 0.5)
        result = predictor.predict_xg(
            player_x=ball_zone.x,
            player_y=ball_zone.y,
            defenders_coords=[(d.position.x, d.position.y) for d in defenders] if defenders else []
        )
        return result

class GoalThreatCalculator:
    """Main class to compute the goal threat"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, x_divisions: int = 15, y_divisions: int = 10):
        self.terrain = Terrain(x_divisions, y_divisions)
        self.movement_calculator = MovementCalculator()
        self.important_players_identifier = ImportantPlayersIdentifier()
        self.config_generator = ConfigurationGenerator(self.movement_calculator)
        self.pass_predictor = PassPredictor()
        self.shot_predictor = ShotPredictor()

    def calculate_threat(self, game_state: GameState, passes_remaining: int = 3):
        """Entry point for the goal threat computation"""
        return self._calculate_threat_recursive(game_state, passes_remaining, ())

    def _calculate_threat_recursive(self, game_state: GameState, passes_remaining: int, path):
        """Compute the goal threat recursively
        Args:
            game_state: Current game state
            passes_remaining: Number of remaining passes to consider
            path: Path traversed so far
            
        Returns:
            float: Goal threat value
            Tuple[Zone]: Optimal path to reach the goal threat
        """
        current_zone = game_state.ball_zone
        optimal_path = (current_zone,)
        
        # Compute the probability of scoring directly
        shot_prob = self.shot_predictor.predict_probability(
            game_state.ball_zone, 
            game_state.goal, 
            game_state.defenders
        )
        
        # If no more passes possible or in an optimal shooting zone, shoot
        if passes_remaining == 0 or game_state.ball_zone in truth_zone:
            return shot_prob, optimal_path
        else:
            # Initialize with the direct shot value
            threat = shot_prob
            optimal_path = (current_zone,)
            
            # Explore the reachable zones
            reachable_zones = self.terrain.get_reachable_zones(game_state.ball_zone)
            for target_zone in reachable_zones:
                # Sample points in the target zone
                candidate_points = self.terrain.sample_points_in_zone(target_zone, num_samples=4)
                best_point = None
                best_pass_prob = 0.0
                
                # Find the best point for the pass
                for point in candidate_points:
                    pass_prob = self.pass_predictor.predict_probability(game_state.ball_zone, point, game_state)
                    if pass_prob > best_pass_prob:
                        best_pass_prob = pass_prob
                        best_point = point

                if best_point:
                    # Generate configurations after the pass
                    configurations = self.config_generator.generate_configurations(
                        game_state, 
                        target_zone, 
                        game_state.defenders, 
                        game_state.teammates, 
                        num_samples=3
                    )

                    # Explore the possible configurations
                    for config in configurations:
                        future_threat, future_path = self._calculate_threat_recursive(
                            config, 
                            passes_remaining - 1,
                            path + (current_zone,)
                        )
                        total_threat = best_pass_prob * future_threat
                        
                        # Update if a better threat is found
                        if total_threat > threat:
                            threat = total_threat
                            optimal_path = (current_zone,) + future_path
    
            return threat, optimal_path
            
    def goal_threat(self, ball_zone: Tuple[float, float], 
                teammates_position: List[Tuple[float, float]], 
                defenders_position: List[Tuple[float, float]], 
                passes_restantes: int = 2) -> Tuple[float, List]:
        """
        Simplified interface to compute the goal threat and the optimal path
        
        Returns:
            Tuple[float, List]: (threat value, optimal path)
        """
        game_state = GameState.from_positions(ball_zone, teammates_position, defenders_position)
        return self.calculate_threat(game_state, passes_restantes)


_default_calculator = None


def goal_threat(ball_zone, teammates_position, defenders_position, passes_restantes=2):
    """Module-level shortcut using a shared GoalThreatCalculator instance."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = GoalThreatCalculator()
    return _default_calculator.goal_threat(
        ball_zone, teammates_position, defenders_position, passes_restantes
    )