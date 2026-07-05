import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import json
from pathlib import Path

class GoAloneSimplePredictor:
    """
    Un prédicteur simple pour évaluer les chances de succès d'une action individuelle ('go alone')
    au football, basé uniquement sur des formules mathématiques (sans ML).
    
    Ce modèle utilise des distributions gaussiennes avec des sigmas paramétrables pour
    modéliser différentes densités (défensive proche, trajectoire, etc.).
    
    Cette version inclut les tirs directs et les tirs après progression, pour une analyse
    complète des situations de tir, basée uniquement sur des facteurs situationnels
    et non sur les caractéristiques des joueurs.
    """
    
    def __init__(self, 
            sigma_close=None,        # Valeur automatique selon normalized_coords
            sigma_trajectory=None,   # Valeur automatique selon normalized_coords
            sigma_angle=np.pi/4,     # En radians, indépendant de la normalisation
            distance_weight=0.25,    
            angle_weight=0.30,       
            close_density_weight=0.25,
            trajectory_density_weight=0.20,
            direct_shot_bonus=0.05,  
            normalized_coords=True,  
            ):
        
        self.normalized_coords = normalized_coords
        
        # Définir les dimensions du terrain
        if normalized_coords:
            self.pitch_length = 1.0
            self.pitch_width = 1.0
            # Valeurs par défaut pour coordonnées normalisées
            self.sigma_close = 0.05 if sigma_close is None else sigma_close
            self.sigma_trajectory = 0.08 if sigma_trajectory is None else sigma_trajectory
        else:
            self.pitch_length = 120.0
            self.pitch_width = 80.0
            # Valeurs par défaut pour coordonnées en mètres
            self.sigma_close = 8.0 if sigma_close is None else sigma_close
            self.sigma_trajectory = 5.0 if sigma_trajectory is None else sigma_trajectory
        
        # ... reste du code inchangé ...
        
        self.sigma_angle = sigma_angle
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.close_density_weight = close_density_weight
        self.trajectory_density_weight = trajectory_density_weight
        self.direct_shot_bonus = direct_shot_bonus
        

    def calculate_close_defensive_density(self, player_x, player_y, defenders_coords):
        """
        Calcule la densité défensive autour du joueur avec une distribution gaussienne.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            defenders_coords: Liste de tuples (x, y) représentant les positions des défenseurs
            
        Returns:
            float: Densité défensive proche (plus elle est élevée, plus il y a de défenseurs proches)
        """
        if not defenders_coords:
            return 0.0
        
        # Calcul de la contribution de chaque défenseur à la densité
        density = 0
        for def_x, def_y in defenders_coords:
            # Distance euclidienne entre le joueur et le défenseur
            dist = np.sqrt((player_x - def_x)**2 + (player_y - def_y)**2)
            
            # Distribution gaussienne: plus le défenseur est proche, plus sa contribution est élevée
            contribution = np.exp(-(dist**2) / (2 * self.sigma_close**2))
            density += contribution
            
        return density
    
    def calculate_trajectory_defensive_density(self, player_x, player_y, goal_x, goal_y, defenders_coords):
        """
        Calcule la densité défensive sur la trajectoire vers le but avec une distribution gaussienne.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            goal_x, goal_y: Coordonnées du but
            defenders_coords: Liste de tuples (x, y) représentant les positions des défenseurs
            
        Returns:
            float: Densité défensive sur la trajectoire (plus elle est élevée, plus la trajectoire est obstruée)
        """
        if not defenders_coords:
            return 0.0
        
        # Vecteur normalisé de la trajectoire vers le but
        trajectory_vector = np.array([goal_x - player_x, goal_y - player_y])
        norm = np.linalg.norm(trajectory_vector)
        if norm == 0:
            return 0.0
        
        trajectory_vector = trajectory_vector / norm
        
        # Calcul de la contribution de chaque défenseur à la densité sur la trajectoire
        density = 0
        for def_x, def_y in defenders_coords:
            # Vecteur du joueur au défenseur
            player_to_def = np.array([def_x - player_x, def_y - player_y])
            
            # Projection du vecteur joueur-défenseur sur la trajectoire
            projection_length = np.dot(player_to_def, trajectory_vector)
            
            # Point de projection sur la trajectoire
            projection_point = np.array([player_x, player_y]) + projection_length * trajectory_vector
            
            # Distance du défenseur à la trajectoire (perpendiculaire)
            dist_to_trajectory = np.linalg.norm(np.array([def_x, def_y]) - projection_point)
            
            # Distance du point de projection au joueur
            dist_along_trajectory = projection_length
            
            # On ne considère que les défenseurs devant le joueur et pas trop loin
            if projection_length > 0 and projection_length < norm:
                # Contribution basée sur la distance perpendiculaire à la trajectoire
                # et la position le long de la trajectoire
                contribution = np.exp(-(dist_to_trajectory**2) / (2 * self.sigma_trajectory**2))
                density += contribution
                
        return density
    
    def calculate_angle_penalty(self, player_x, player_y, goal_x, goal_y):
        """
        Calcule une pénalité basée sur l'angle entre le joueur et le but.
        Un angle de 0 est idéal (joueur face au but), une pénalité plus élevée
        est appliquée pour les angles plus grands.
        
        Version modifiée pour être moins sévère avec les angles difficiles.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            goal_x, goal_y: Coordonnées du but
            
        Returns:
            float: Valeur entre 0 et 1, où 0 est l'angle idéal et 1 est la pénalité maximale
        """
        # Vecteur du joueur au but
        to_goal = np.array([goal_x - player_x, goal_y - player_y])
        
        # Normaliser le vecteur
        if np.linalg.norm(to_goal) == 0:
            return 1.0  # Cas où le joueur est exactement sur le but (peu probable)
        
        to_goal = to_goal / np.linalg.norm(to_goal)
        
        # Définir l'axe principal du terrain (selon coordonnées normalisées ou non)
        if self.normalized_coords:
            # En coordonnées normalisées, l'axe x est l'axe principal
            main_axis = np.array([1.0, 0.0])
        else:
            # En coordonnées standard (en mètres), l'axe x est l'axe principal
            main_axis = np.array([1.0, 0.0])
        
        # L'angle est calculé par rapport à l'axe principal du terrain
        cos_angle = np.dot(to_goal, main_axis)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Version moins sévère: utiliser une fonction plus douce
        # Au lieu d'une gaussienne, utiliser une fonction qui pénalise moins les angles intermédiaires
        # Cette formule réduit la pénalité pour les angles jusqu'à ~60 degrés
        penalty = (angle / np.pi) ** 1.5  # Exposant < 2 pour être moins sévère
        
        return penalty

    def calculate_distance_factor(self, player_x, player_y, goal_x, goal_y):
        """
        Calcule un facteur basé sur la distance au but.
        Une distance plus courte donne une meilleure chance de succès.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            goal_x, goal_y: Coordonnées du but
            
        Returns:
            float: Valeur entre 0 et 1, où 1 indique une distance idéale (proche du but)
        """
        # Distance euclidienne au but
        dist = np.sqrt((player_x - goal_x)**2 + (player_y - goal_y)**2)
        
        # Normaliser par rapport à la diagonale du terrain (distance maximale possible)
        max_dist = np.sqrt(self.pitch_length**2 + self.pitch_width**2)
        normalized_dist = dist / max_dist
        
        # Convertir en facteur: 1 pour distance=0, diminue exponentiellement avec la distance
        # Utiliser une formule qui donne une valeur raisonnable à mi-terrain
        factor = np.exp(-3 * normalized_dist)
        
        return factor
    
    def predict_success_probability(self, player_x, player_y, defenders_coords, goal_x=None, goal_y=None, 
                       is_direct_shot=False):
        """
        Prédit la probabilité de succès d'une action "go alone" en utilisant une approche multiplicative
        avec des exponentielles pour mieux représenter l'impact des défenseurs.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            defenders_coords: Liste de tuples (x, y) représentant les positions des défenseurs
            goal_x, goal_y: Coordonnées du but (par défaut, au bout du terrain)
            is_direct_shot: Booléen indiquant si c'est un tir direct (sans dribble)
            
        Returns:
            dict: Dictionnaire contenant la probabilité et les facteurs détaillés
        """
        # Définir les coordonnées du but si non spécifiées
        if goal_x is None:
            goal_x = self.pitch_length  # Extrémité du terrain
        if goal_y is None:
            goal_y = self.pitch_width / 2  # Centre de la largeur
        
        # Calculer tous les composants
        close_density = self.calculate_close_defensive_density(player_x, player_y, defenders_coords)
        trajectory_density = self.calculate_trajectory_defensive_density(
            player_x, player_y, goal_x, goal_y, defenders_coords
        )
        angle_penalty = self.calculate_angle_penalty(player_x, player_y, goal_x, goal_y)
        distance_factor = self.calculate_distance_factor(player_x, player_y, goal_x, goal_y)
        
        # --- Approche multiplicative avec exponentielles ---

        # Probabilité de base (valeur optimale théorique dans des conditions parfaites)

        # Facteur de distance
        distance_component = distance_factor ** 1.0  # Inchangé

        # Facteur d'angle (1 - penalty)
        angle_factor = 1 - angle_penalty
        angle_component = angle_factor  # Inchangé

        # Facteurs de densité défensive - RÉDUIRE DAVANTAGE les coefficients
        close_density_component = close_density     # Coefficient réduit (était -1.0)
        trajectory_density_component = trajectory_density

        # Combinaison multiplicative de tous les facteurs
        probability = ( 
            distance_component * 
            angle_component * 
            close_density_component * 
            trajectory_density_component
        )
            
        # Bonus pour tirs directs (moins de risque d'interception)
        if is_direct_shot:
            probability *= 1.15  # Bonus de 15% pour les tirs directs
        
        # S'assurer que la probabilité est entre 0 et 1
        probability = max(0.0, min(0.95, probability))  # Plafonné à 0.95 pour éviter les certitudes
        
        return {
            'probability': probability,
            'factors': {
                'distance_factor': distance_component,
                'angle_factor': angle_component,
                'close_density_component': close_density_component,
                'trajectory_density_component': trajectory_density_component,
                'is_direct_shot': is_direct_shot,
                'actual_xg': None  # Placeholder pour l'actual_xg, à remplir lors de l'optimisation
            }
        }
    
    def visualize_prediction(self, player_x, player_y, defenders_coords, goal_x=120, goal_y=40, 
                            pitch_length=120, pitch_width=80, show_densities=True):
        """
        Visualise la situation avec le joueur, les défenseurs, et la probabilité prédite.
        
        Args:
            player_x, player_y: Coordonnées du joueur avec le ballon
            defenders_coords: Liste de tuples (x, y) représentant les positions des défenseurs
            goal_x, goal_y: Coordonnées du but
            pitch_length, pitch_width: Dimensions du terrain
            show_densities: Afficher ou non les cartes de chaleur des densités
        """
        if pitch_length is None:
            pitch_length = self.pitch_length
        if pitch_width is None:
            pitch_width = self.pitch_width
        
        # Utiliser les coordonnées du but de l'instance si non spécifiées
        if goal_x is None:
            goal_x = pitch_length  # Extrémité du terrain
        if goal_y is None:
            goal_y = pitch_width / 2  # Centre de la largeur
        
        # Calculer la probabilité
        result = self.predict_success_probability(player_x, player_y, defenders_coords, goal_x, goal_y)
        probability = result['probability']  # Extraire la probabilité du dictionnaire
        factors = result['factors']  # Extraire les facteurs pour information supplémentaire
        
        # Créer la figure
        if show_densities:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            ax_pitch = axs[0]
            ax_close = axs[1]
            ax_trajectory = axs[2]
        else:
            fig, ax_pitch = plt.subplots(figsize=(10, 7))
        
        # Dessiner le terrain
        ax_pitch.set_xlim(0, pitch_length)
        ax_pitch.set_ylim(0, pitch_width)
        ax_pitch.add_patch(plt.Rectangle((0, 0), pitch_length, pitch_width, fill=False, color='green'))
        
        # Dessiner le but
        goal_width = 7.32  # Largeur standard d'un but de football en mètres
        ax_pitch.add_patch(plt.Rectangle((goal_x - 1, goal_y - goal_width/2), 1, goal_width, 
                                        fill=True, color='gray'))
        
        # Dessiner le joueur
        ax_pitch.scatter(player_x, player_y, color='blue', s=100, label='Joueur')
        
        # Dessiner la trajectoire vers le but
        ax_pitch.arrow(player_x, player_y, goal_x - player_x, goal_y - player_y, 
                    head_width=2, head_length=2, fc='blue', ec='blue', alpha=0.5)
        
        # Dessiner les défenseurs
        if defenders_coords:
            defenders_x, defenders_y = zip(*defenders_coords)
            ax_pitch.scatter(defenders_x, defenders_y, color='red', s=80, label='Défenseurs')
        
        # Ajouter le titre avec la probabilité
        ax_pitch.set_title(f'Probabilité de succès: {probability:.2f}', fontsize=14)
        
        # Ajouter une annotation avec les facteurs
        # Ajouter une annotation avec les facteurs
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
        
        # Si demandé, afficher les cartes de chaleur des densités
        if show_densities and defenders_coords:
            # Créer une grille pour les cartes de chaleur
            x_grid = np.linspace(0, pitch_length, 100)
            y_grid = np.linspace(0, pitch_width, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Carte de chaleur pour la densité proche
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
            
            # Carte de chaleur pour la densité sur la trajectoire
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


# Exemple d'utilisation avec coordonnées normalisées
if __name__ == "__main__":
    # Créer une instance du prédicteur avec des coordonnées normalisées
    predictor = GoAloneSimplePredictor(
        normalized_coords=True,
        sigma_close=0.03,        # 5% de la largeur du terrain
        sigma_trajectory=0.08    # 8% de la largeur du terrain
    )
    
    print("=== EXEMPLES DE SITUATIONS AVEC COORDONNÉES NORMALISÉES (0-1) ===")
    # Exemple 6: Situation réaliste avec 11 défenseurs (formation 4-4-2)
    print("\nExemple 6: Situation réaliste avec 11 défenseurs (formation 4-4-2)")
    player_x, player_y = 0.75, 0.5  # Attaquant à environ 25m du but
    defenders_coords = [
        # Gardien
        (0.98, 0.5),  # Gardien de but
        
        # Défense à 4
        (0.90, 0.3),  # Arrière latéral droit
        (0.90, 0.43), # Défenseur central droit
        (0.90, 0.57), # Défenseur central gauche
        (0.90, 0.7),  # Arrière latéral gauche
        
        # Milieu à 4
        (0.82, 0.25), # Milieu droit
        (0.80, 0.40), # Milieu central droit
        (0.80, 0.60), # Milieu central gauche
        (0.82, 0.75), # Milieu gauche
        
        # Attaquants à 2
        (0.65, 0.45), # Attaquant droit (en position de repli défensif)
        (0.65, 0.55)  # Attaquant gauche (en position de repli défensif)
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

    # Exemple 7: Percée individuelle dans une défense organisée
    print("\nExemple 7: Percée individuelle dans une défense organisée")
    player_x, player_y = 0.80, 0.38  # Attaquant ayant percé sur le côté droit
    defenders_coords = [
        # Gardien
        (0.98, 0.5),  # Gardien de but
        
        # Défenseurs à proximité (dépassés par la percée)
        (0.75, 0.40), # Défenseur dépassé
        (0.85, 0.45), # Défenseur en couverture
        (0.90, 0.55), # Défenseur central
        
        # Reste de la défense
        (0.88, 0.65), # Défenseur central gauche
        (0.85, 0.75), # Arrière latéral gauche
        
        # Milieu à 4 (en repli défensif)
        (0.70, 0.25), # Milieu droit
        (0.65, 0.45), # Milieu central
        (0.65, 0.60), # Milieu central gauche
        (0.70, 0.80), # Milieu gauche
        
        # Attaquant en repli
        (0.60, 0.50)  # Attaquant
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

    # Exemple 8: Face-à-face avec le gardien après avoir battu la défense
    print("\nExemple 8: Face-à-face avec le gardien après avoir battu la défense")
    player_x, player_y = 0.93, 0.48  # Attaquant en position idéale face au gardien
    defenders_coords = [
        # Gardien
        (0.98, 0.5),  # Gardien de but
        
        # Défenseurs battus en train de revenir
        (0.88, 0.45), # Défenseur en retard
        (0.85, 0.55), # Défenseur en retard
        
        # Reste de l'équipe (trop loin pour interférer directement)
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
    # Exemple 1: Position très favorable (quasi-penalty)
    print("\nExemple 1: Position très favorable (quasi-penalty)")
    player_x, player_y = 0.92, 0.5  # Très proche du but, position centrale
    defenders_coords = [
        (0.85, 0.65),  # Un défenseur loin sur le côté
        (0.88, 0.30),  # Un autre défenseur loin de l'autre côté
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
    
    # Exemple 2: Position très difficile - bloc défensif massif
    print("\nExemple 2: Position très difficile - bloc défensif massif")
    player_x, player_y = 0.8, 0.5  # Distance moyenne
    defenders_coords = [
        (0.82, 0.48),  # Défenseur 1 juste devant
        (0.85, 0.52),  # Défenseur 2
        (0.88, 0.45),  # Défenseur 3
        (0.86, 0.50),  # Défenseur 4 - aligné directement avec but
        (0.90, 0.47),  # Défenseur 5
        (0.92, 0.53),  # Défenseur 6
        (0.84, 0.55),  # Défenseur 7
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
    
    # Exemple 3: Tir depuis le milieu de terrain
    print("\nExemple 3: Tir depuis le milieu de terrain")
    player_x, player_y = 0.5, 0.5  # Milieu de terrain
    defenders_coords = [
        (0.6, 0.48),   # Quelques défenseurs entre le joueur et le but
        (0.7, 0.52),
        (0.8, 0.45),
    ]
    result = predictor.predict_success_probability(
        player_x, player_y, defenders_coords,
        is_direct_shot=True  # Tir direct
    )
    probability = result['probability']
    print(f"Probabilité de succès (tir de loin): {probability:.4f}")
    predictor.visualize_prediction(
        player_x, player_y, defenders_coords,
        goal_x=1.0, goal_y=0.5,
        pitch_length=1.0, pitch_width=1.0,
        show_densities=True
    )
    
    # Exemple 4: Position latérale avec angle difficile
    print("\nExemple 4: Position latérale avec angle difficile")
    player_x, player_y = 0.9, 0.15  # Proche du but mais angle très fermé
    defenders_coords = [
        (0.92, 0.20),  # Un défenseur proche
        (0.95, 0.30),  # Un défenseur un peu plus loin
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
    
    # Exemple 5: Position idéale mais totalement encerclé
    print("\nExemple 5: Position idéale mais totalement encerclé")
    player_x, player_y = 0.85, 0.5  # Position idéale
    defenders_coords = [
        (0.87, 0.5),   # Défenseur directement devant
        (0.85, 0.45),  # Défenseur sur le côté
        (0.85, 0.55),  # Défenseur sur l'autre côté
        (0.83, 0.5),   # Défenseur juste derrière
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