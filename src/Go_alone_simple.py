import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import json
from pathlib import Path

class GoAloneSimplePredictor:
    """
    A simple predictor to evaluate the success chances of an individual action ('go alone')
    in football, based only on mathematical formulas (no ML).
    
    This model uses Gaussian distributions with tunable sigmas to
    model different densities (close defensive, trajectory, etc.).
    
    This version includes direct shots and shots after progression, for a complete
    analysis of shooting situations, based only on situational factors
    and not on player characteristics.
    """
    
    def __init__(self, 
            sigma_close=None,        # Automatic value depending on normalized_coords
            sigma_trajectory=None,   # Automatic value depending on normalized_coords
            sigma_angle=np.pi/4,     # In radians, independent of normalization
            distance_weight=0.25,    
            angle_weight=0.30,       
            close_density_weight=0.25,
            trajectory_density_weight=0.20,
            direct_shot_bonus=0.05,  
            normalized_coords=True,  
            ):
        
        self.normalized_coords = normalized_coords
        
        # Define the pitch dimensions
        if normalized_coords:
            self.pitch_length = 1.0
            self.pitch_width = 1.0
            # Default values for normalized coordinates
            self.sigma_close = 0.05 if sigma_close is None else sigma_close
            self.sigma_trajectory = 0.08 if sigma_trajectory is None else sigma_trajectory
        else:
            self.pitch_length = 120.0
            self.pitch_width = 80.0
            # Default values for coordinates in meters
            self.sigma_close = 8.0 if sigma_close is None else sigma_close
            self.sigma_trajectory = 5.0 if sigma_trajectory is None else sigma_trajectory
        
        # ... rest of the code unchanged ...
        
        self.sigma_angle = sigma_angle
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.close_density_weight = close_density_weight
        self.trajectory_density_weight = trajectory_density_weight
        self.direct_shot_bonus = direct_shot_bonus
        

    def calculate_close_defensive_density(self, player_x, player_y, defenders_coords):
        """
        Compute the defensive density around the player with a Gaussian distribution.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            defenders_coords: List of (x, y) tuples representing the defenders' positions
            
        Returns:
            float: Close defensive density (the higher it is, the more defenders are nearby)
        """
        if not defenders_coords:
            return 0.0
        
        # Compute each defender's contribution to the density
        density = 0
        for def_x, def_y in defenders_coords:
            # Euclidean distance between the player and the defender
            dist = np.sqrt((player_x - def_x)**2 + (player_y - def_y)**2)
            
            # Gaussian distribution: the closer the defender, the higher its contribution
            contribution = np.exp(-(dist**2) / (2 * self.sigma_close**2))
            density += contribution
            
        return density
    
    def calculate_trajectory_defensive_density(self, player_x, player_y, goal_x, goal_y, defenders_coords):
        """
        Compute the defensive density along the trajectory toward the goal with a Gaussian distribution.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            goal_x, goal_y: Goal coordinates
            defenders_coords: List of (x, y) tuples representing the defenders' positions
            
        Returns:
            float: Trajectory defensive density (the higher it is, the more obstructed the trajectory)
        """
        if not defenders_coords:
            return 0.0
        
        # Normalized vector of the trajectory toward the goal
        trajectory_vector = np.array([goal_x - player_x, goal_y - player_y])
        norm = np.linalg.norm(trajectory_vector)
        if norm == 0:
            return 0.0
        
        trajectory_vector = trajectory_vector / norm
        
        # Compute each defender's contribution to the trajectory density
        density = 0
        for def_x, def_y in defenders_coords:
            # Vector from the player to the defender
            player_to_def = np.array([def_x - player_x, def_y - player_y])
            
            # Projection of the player-defender vector onto the trajectory
            projection_length = np.dot(player_to_def, trajectory_vector)
            
            # Projection point on the trajectory
            projection_point = np.array([player_x, player_y]) + projection_length * trajectory_vector
            
            # Distance from the defender to the trajectory (perpendicular)
            dist_to_trajectory = np.linalg.norm(np.array([def_x, def_y]) - projection_point)
            
            # Distance from the projection point to the player
            dist_along_trajectory = projection_length
            
            # Only consider defenders in front of the player and not too far
            if projection_length > 0 and projection_length < norm:
                # Contribution based on the perpendicular distance to the trajectory
                # and the position along the trajectory
                contribution = np.exp(-(dist_to_trajectory**2) / (2 * self.sigma_trajectory**2))
                density += contribution
                
        return density
    
    def calculate_angle_penalty(self, player_x, player_y, goal_x, goal_y):
        """
        Compute a penalty based on the angle between the player and the goal.
        An angle of 0 is ideal (player facing the goal), a higher penalty
        is applied for larger angles.
        
        Modified version to be less severe with difficult angles.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            goal_x, goal_y: Goal coordinates
            
        Returns:
            float: Value between 0 and 1, where 0 is the ideal angle and 1 is the maximum penalty
        """
        # Vector from the player to the goal
        to_goal = np.array([goal_x - player_x, goal_y - player_y])
        
        # Normalize the vector
        if np.linalg.norm(to_goal) == 0:
            return 1.0  # Case where the player is exactly on the goal (unlikely)
        
        to_goal = to_goal / np.linalg.norm(to_goal)
        
        # Define the main axis of the pitch (depending on normalized coordinates or not)
        if self.normalized_coords:
            # In normalized coordinates, the x axis is the main axis
            main_axis = np.array([1.0, 0.0])
        else:
            # In standard coordinates (in meters), the x axis is the main axis
            main_axis = np.array([1.0, 0.0])
        
        # The angle is computed relative to the main axis of the pitch
        cos_angle = np.dot(to_goal, main_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Less severe version: use a smoother function
        # Instead of a Gaussian, use a function that penalizes intermediate angles less
        # This formula reduces the penalty for angles up to ~60 degrees
        penalty = (angle / np.pi) ** 1.5  # Exponent < 2 to be less severe
        
        return penalty

    def calculate_distance_factor(self, player_x, player_y, goal_x, goal_y):
        """
        Compute a factor based on the distance to the goal.
        A shorter distance gives a better chance of success.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            goal_x, goal_y: Goal coordinates
            
        Returns:
            float: Value between 0 and 1, where 1 indicates an ideal distance (close to goal)
        """
        # Euclidean distance to the goal
        dist = np.sqrt((player_x - goal_x)**2 + (player_y - goal_y)**2)
        
        # Normalize with respect to the pitch diagonal (maximum possible distance)
        max_dist = np.sqrt(self.pitch_length**2 + self.pitch_width**2)
        normalized_dist = dist / max_dist
        
        # Convert into a factor: 1 for distance=0, decreases exponentially with distance
        # Use a formula that gives a reasonable value at midfield
        factor = np.exp(-3 * normalized_dist)
        
        return factor
    
    def predict_success_probability(self, player_x, player_y, defenders_coords, goal_x=None, goal_y=None, 
                       is_direct_shot=False):
        """
        Predict the success probability of a "go alone" action using a multiplicative approach
        with exponentials to better represent the impact of defenders.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            defenders_coords: List of (x, y) tuples representing the defenders' positions
            goal_x, goal_y: Goal coordinates (by default, at the end of the pitch)
            is_direct_shot: Boolean indicating whether it is a direct shot (without dribbling)
            
        Returns:
            dict: Dictionary containing the probability and the detailed factors
        """
        # Define the goal coordinates if not specified
        if goal_x is None:
            goal_x = self.pitch_length  # End of the pitch
        if goal_y is None:
            goal_y = self.pitch_width / 2  # Center of the width
        
        # Compute all components
        close_density = self.calculate_close_defensive_density(player_x, player_y, defenders_coords)
        trajectory_density = self.calculate_trajectory_defensive_density(
            player_x, player_y, goal_x, goal_y, defenders_coords
        )
        angle_penalty = self.calculate_angle_penalty(player_x, player_y, goal_x, goal_y)
        distance_factor = self.calculate_distance_factor(player_x, player_y, goal_x, goal_y)
        
        # --- Multiplicative approach with exponentials ---

        # Base probability (theoretical optimal value in perfect conditions)

        # Distance factor
        distance_component = distance_factor ** 1.0  # Unchanged

        # Angle factor (1 - penalty)
        angle_factor = 1 - angle_penalty
        angle_component = angle_factor  # Unchanged

        # Defensive density factors - REDUCE the coefficients FURTHER
        close_density_component = close_density     # Reduced coefficient (was -1.0)
        trajectory_density_component = trajectory_density

        # Multiplicative combination of all factors
        probability = ( 
            distance_component * 
            angle_component * 
            close_density_component * 
            trajectory_density_component
        )
            
        # Bonus for direct shots (less interception risk)
        if is_direct_shot:
            probability *= 1.15  # 15% bonus for direct shots
        
        # Make sure the probability is between 0 and 1
        probability = max(0.0, min(0.95, probability))  # Capped at 0.95 to avoid certainties
        
        return {
            'probability': probability,
            'factors': {
                'distance_factor': distance_component,
                'angle_factor': angle_component,
                'close_density_component': close_density_component,
                'trajectory_density_component': trajectory_density_component,
                'is_direct_shot': is_direct_shot,
                'actual_xg': None  # Placeholder for actual_xg, to be filled during optimization
            }
        }
    
    def visualize_prediction(self, player_x, player_y, defenders_coords, goal_x=120, goal_y=40, 
                            pitch_length=120, pitch_width=80, show_densities=True):
        """
        Visualize the situation with the player, the defenders, and the predicted probability.
        
        Args:
            player_x, player_y: Coordinates of the player with the ball
            defenders_coords: List of (x, y) tuples representing the defenders' positions
            goal_x, goal_y: Goal coordinates
            pitch_length, pitch_width: Pitch dimensions
            show_densities: Whether or not to show the density heatmaps
        """
        if pitch_length is None:
            pitch_length = self.pitch_length
        if pitch_width is None:
            pitch_width = self.pitch_width
        
        # Use the instance's goal coordinates if not specified
        if goal_x is None:
            goal_x = pitch_length  # End of the pitch
        if goal_y is None:
            goal_y = pitch_width / 2  # Center of the width
        
        # Compute the probability
        result = self.predict_success_probability(player_x, player_y, defenders_coords, goal_x, goal_y)
        probability = result['probability']  # Extract the probability from the dictionary
        factors = result['factors']  # Extract the factors for additional information
        
        # Create the figure
        if show_densities:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            ax_pitch = axs[0]
            ax_close = axs[1]
            ax_trajectory = axs[2]
        else:
            fig, ax_pitch = plt.subplots(figsize=(10, 7))
        
        # Draw the pitch
        ax_pitch.set_xlim(0, pitch_length)
        ax_pitch.set_ylim(0, pitch_width)
        ax_pitch.add_patch(plt.Rectangle((0, 0), pitch_length, pitch_width, fill=False, color='green'))
        
        # Draw the goal
        goal_width = 7.32  # Standard width of a football goal in meters
        ax_pitch.add_patch(plt.Rectangle((goal_x - 1, goal_y - goal_width/2), 1, goal_width, 
                                        fill=True, color='gray'))
        
        # Draw the player
        ax_pitch.scatter(player_x, player_y, color='blue', s=100, label='Joueur')
        
        # Draw the trajectory toward the goal
        ax_pitch.arrow(player_x, player_y, goal_x - player_x, goal_y - player_y, 
                    head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.5)
        
        # Draw the defenders
        if defenders_coords:
            defenders_x, defenders_y = zip(*defenders_coords)
            ax_pitch.scatter(defenders_x, defenders_y, color='red', s=80, label='Défenseurs')
        
        # Add the title with the probability
        ax_pitch.set_title(f'Probabilité de succès: {probability:.2f}', fontsize=14)
        
        # Add an annotation with the factors
        # Add an annotation with the factors
        factor_text = "\n".join([
            f"Distance: {factors['distance_factor']:.2f}",
            f"Angle: {factors['angle_factor']:.2f}",
            f"Densité proche: {factors['close_density_component']:.2f}",
            f"Densité trajectoire: {factors['trajectory_density_component']:.2f}",
            f"Tir direct: {'Oui' if factors['is_direct_shot'] else 'Non'}"
        ])
        ax_pitch.text(0.05, 0.95, factor_text, transform=ax_pitch.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        ax_pitch.legend()
        
        # If requested, show the density heatmaps
        if show_densities and defenders_coords:
            # Create a grid for the heatmaps
            x_grid = np.linspace(0, pitch_length, 100)
            y_grid = np.linspace(0, pitch_width, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Heatmap for the close density
            close_density = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    close_density[i, j] = self.calculate_close_defensive_density(
                        X[i, j], Y[i, j], defenders_coords
                    )
            
            im_close = ax_close.imshow(close_density, extent=[0, pitch_length, 0, pitch_width], 
                                    origin='lower', cmap='hot_r')
            ax_close.set_title('Densité défensive proche')
            ax_close.scatter(player_x, player_y, color='blue', s=100)
            if defenders_coords:
                defenders_x, defenders_y = zip(*defenders_coords)
                ax_close.scatter(defenders_x, defenders_y, color='red', s=80)
            plt.colorbar(im_close, ax=ax_close)
            
            # Heatmap for the trajectory density
            trajectory_density = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    trajectory_density[i, j] = self.calculate_trajectory_defensive_density(
                        X[i, j], Y[i, j], goal_x, goal_y, defenders_coords
                    )
            
            im_trajectory = ax_trajectory.imshow(trajectory_density, 
                                            extent=[0, pitch_length, 0, pitch_width], 
                                            origin='lower', cmap='hot_r')
            ax_trajectory.set_title('Densité sur la trajectoire vers le but')
            ax_trajectory.scatter(player_x, player_y, color='blue', s=100)
            ax_trajectory.scatter(goal_x, goal_y, color='green', s=100, marker='s')
            ax_trajectory.arrow(player_x, player_y, goal_x - player_x, goal_y - player_y, 
                            head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.5)
            if defenders_coords:
                defenders_x, defenders_y = zip(*defenders_coords)
                ax_trajectory.scatter(defenders_x, defenders_y, color='red', s=80)
            plt.colorbar(im_trajectory, ax=ax_trajectory)
        
        plt.tight_layout()
        plt.show()


# Example usage with normalized coordinates
if __name__ == "__main__":
    # Create an instance of the predictor with normalized coordinates
    predictor = GoAloneSimplePredictor(
        normalized_coords=True,
        sigma_close=0.03,        # 5% of the pitch width
        sigma_trajectory=0.08    # 8% of the pitch width
    )
    
    print("=== EXEMPLES DE SITUATIONS AVEC COORDONNÉES NORMALISÉES (0-1) ===")
    # Example 6: Realistic situation with 11 defenders (4-4-2 formation)
    print("\nExemple 6: Situation réaliste avec 11 défenseurs (formation 4-4-2)")
    player_x, player_y = 0.75, 0.5  # Attacker about 25m from goal
    defenders_coords = [
        # Goalkeeper
        (0.98, 0.5),  # Goalkeeper
        
        # Back four
        (0.90, 0.3),  # Right fullback
        (0.90, 0.43), # Right center-back
        (0.90, 0.57), # Left center-back
        (0.90, 0.7),  # Left fullback
        
        # Midfield four
        (0.82, 0.25), # Right midfielder
        (0.80, 0.40), # Right central midfielder
        (0.80, 0.60), # Left central midfielder
        (0.82, 0.75), # Left midfielder
        
        # Two forwards
        (0.65, 0.45), # Right forward (in defensive recovery position)
        (0.65, 0.55)  # Left forward (in defensive recovery position)
    ]

    result = predictor.predict_success_probability(player_x, player_y, defenders_coords)
    probability = result['probability']
    print(f"Probabilité de succès (face à 11 défenseurs): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )

    # Example 7: Individual breakthrough against an organized defense
    print("\nExemple 7: Percée individuelle dans une défense organisée")
    player_x, player_y = 0.80, 0.38  # Attacker who broke through on the right side
    defenders_coords = [
        # Goalkeeper
        (0.98, 0.5),  # Goalkeeper
        
        # Nearby defenders (beaten by the breakthrough)
        (0.75, 0.40), # Beaten defender
        (0.85, 0.45), # Covering defender
        (0.90, 0.55), # Center-back
        
        # Rest of the defense
        (0.88, 0.65), # Left center-back
        (0.85, 0.75), # Left fullback
        
        # Midfield four (in defensive recovery)
        (0.70, 0.25), # Right midfielder
        (0.65, 0.45), # Central midfielder
        (0.65, 0.60), # Left central midfielder
        (0.70, 0.80), # Left midfielder
        
        # Recovering forward
        (0.60, 0.50)  # Forward
    ]

    result = predictor.predict_success_probability(player_x, player_y, defenders_coords)
    probability = result['probability']
    print(f"Probabilité de succès (percée individuelle): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )

    # Example 8: One-on-one with the keeper after beating the defense
    print("\nExemple 8: Face-à-face avec le gardien après avoir battu la défense")
    player_x, player_y = 0.93, 0.48  # Attacker in an ideal position facing the keeper
    defenders_coords = [
        # Goalkeeper
        (0.98, 0.5),  # Goalkeeper
        
        # Beaten defenders recovering
        (0.88, 0.45), # Trailing defender
        (0.85, 0.55), # Trailing defender
        
        # Rest of the team (too far to interfere directly)
        (0.80, 0.30),
        (0.78, 0.42),
        (0.75, 0.65),
        (0.82, 0.75),
        (0.70, 0.25),
        (0.65, 0.50),
        (0.68, 0.60),
        (0.60, 0.45)
    ]

    result = predictor.predict_success_probability(player_x, player_y, defenders_coords)
    probability = result['probability']
    print(f"Probabilité de succès (face-à-face avec le gardien): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    # Example 1: Very favorable position (near-penalty)
    print("\nExemple 1: Position très favorable (quasi-penalty)")
    player_x, player_y = 0.92, 0.5  # Very close to goal, central position
    defenders_coords = [
        (0.85, 0.65),  # A defender far out on the side
        (0.88, 0.30),  # Another defender far on the other side
    ]
    result = predictor.predict_success_probability(player_x, player_y, defenders_coords)
    probability = result['probability']
    print(f"Probabilité de succès: {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    
    # Example 2: Very difficult position - massive defensive block
    print("\nExemple 2: Position très difficile - bloc défensif massif")
    player_x, player_y = 0.8, 0.5  # Medium distance
    defenders_coords = [
        (0.82, 0.48),  # Defender 1 right in front
        (0.85, 0.52),  # Defender 2
        (0.88, 0.45),  # Defender 3
        (0.86, 0.50),  # Defender 4 - aligned directly with goal
        (0.90, 0.47),  # Defender 5
        (0.92, 0.53),  # Defender 6
        (0.84, 0.55),  # Defender 7
    ]
    result = predictor.predict_success_probability(player_x, player_y, defenders_coords)
    probability = result['probability']
    print(f"Probabilité de succès: {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    
    # Example 3: Shot from midfield
    print("\nExemple 3: Tir depuis le milieu de terrain")
    player_x, player_y = 0.5, 0.5  # Midfield
    defenders_coords = [
        (0.6, 0.48),   # A few defenders between the player and the goal
        (0.7, 0.52),
        (0.8, 0.45),
    ]
    result = predictor.predict_success_probability(
        player_x, player_y, defenders_coords,
        is_direct_shot=True  # Direct shot
    )
    probability = result['probability']
    print(f"Probabilité de succès (tir de loin): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    
    # Example 4: Wide position with a difficult angle
    print("\nExemple 4: Position latérale avec angle difficile")
    player_x, player_y = 0.9, 0.15  # Close to goal but very tight angle
    defenders_coords = [
        (0.92, 0.20),  # A nearby defender
        (0.95, 0.30),  # A defender a bit further away
    ]
    result = predictor.predict_success_probability(
        player_x, player_y, defenders_coords
    )
    probability = result['probability']
    print(f"Probabilité de succès (angle difficile): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    
    # Example 5: Ideal position but completely surrounded
    print("\nExemple 5: Position idéale mais totalement encerclé")
    player_x, player_y = 0.85, 0.5  # Ideal position
    defenders_coords = [
        (0.87, 0.5),   # Defender directly in front
        (0.85, 0.45),  # Defender on the side
        (0.85, 0.55),  # Defender on the other side
        (0.83, 0.5),   # Defender just behind
    ]
    result = predictor.predict_success_probability(
        player_x, player_y, defenders_coords
    )
    probability = result['probability']
    print(f"Probabilité de succès (encerclé): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )