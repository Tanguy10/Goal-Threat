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
    """Crée un terrain de football standardisé (proportions normalisées)"""
    # Terrain (rectangle vert)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                           edgecolor='white', facecolor='#538032', alpha=0.8)
    ax.add_patch(rect)
    
    # Ligne médiane
    plt.plot([0.5, 0.5], [0, 1], color='white', linewidth=2)
    
    # Cercle central
    circle = plt.Circle((0.5, 0.5), 0.1, color='white', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Surface de réparation (gauche)
    rect = patches.Rectangle((0, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Surface de but (gauche)
    rect = patches.Rectangle((0, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Surface de réparation (droite)
    rect = patches.Rectangle((0.82, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Surface de but (droite)
    rect = patches.Rectangle((0.94, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Ajuster les paramètres du graphique
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def plot_pass_success_probability(ax, origin_x, origin_y, target_x, target_y, prob, annotate=True):
    """Affiche une passe et sa probabilité de réussite"""
    # Tracer la passe (flèche)
    arrow = ax.arrow(origin_x, origin_y, target_x-origin_x, target_y-origin_y, 
                   head_width=0.02, head_length=0.03, fc='black', ec='black',
                   length_includes_head=True, alpha=0.7)
    
    # Afficher la probabilité
    if annotate:
        mid_x = (origin_x + target_x) / 2
        mid_y = (origin_y + target_y) / 2
        ax.annotate(f"{prob:.0%}", (mid_x, mid_y), 
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                  fontsize=10, ha='center', va='center')
    
    return arrow

def plot_player(ax, x, y, team='home', label=None):
    """Affiche un joueur sur le terrain"""
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
    """Charge les hyperparamètres optimisés"""
    # Si le prédicteur est déjà chargé, utiliser ses paramètres
    if 'predictor' in globals() and hasattr(predictor, 'get_optimal_params'):
        params = predictor.get_optimal_params()
        if params:
            print("✅ Hyperparamètres chargés depuis le prédicteur")
            return params
    
    # Sinon, essayer de charger depuis le fichier
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
    """Affiche les hyperparamètres actuellement utilisés"""
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
    Calcule toutes les features d'une passe en utilisant les hyperparamètres optimisés par Optuna
    """
    # ⭐ CHARGER LES HYPERPARAMÈTRES OPTIMAUX D'OPTUNA
    best_params = load_best_hyperparameters()
    
    # Utiliser les valeurs optimisées ou les paramètres par défaut
    sigma_optimal = best_params.get('sigma', 0.05)
    seuil_trajectoire_optimal = best_params.get('seuil_trajectoire', 0.02)
    
    # ⭐ UTILISER LES HYPERPARAMÈTRES OPTIMAUX
    # Adversaires proches du départ (densité pondérée avec sigma optimal)
    nb_adv_proches_depart = densite_adversaires_ponderee(
        x_passeur, y_passeur, adv_positions, 
        sigma=sigma_optimal  # ← Valeur optimisée par Optuna
    )
    
    # Adversaires sur la trajectoire (avec seuil optimal)
    nb_adv_trajectoire = nb_adv_trajectoire_coords(
        x_passeur, y_passeur, x_cible, y_cible, adv_positions, 
        seuil_trajectoire=seuil_trajectoire_optimal  # ← Valeur optimisée par Optuna
    )

    # Coéquipiers sur la trajectoire
    nb_coequipiers_trajectoire = nb_adv_trajectoire_coords(
        x_passeur, y_passeur, x_cible, y_cible, teammate_positions, 
        seuil_trajectoire=seuil_trajectoire_optimal  # ← Valeur optimisée par Optuna
    )
    
    # Adversaires proches de l'arrivée (densité pondérée avec sigma optimal)
    nb_adv_proches_arrivee = densite_adversaires_ponderee(
        x_cible, y_cible, adv_positions, 
        sigma=sigma_optimal  # ← Valeur optimisée par Optuna
    )
    
    # Coéquipiers proches de l'arrivée
    nb_coequipiers_proches_arrivee = densite_adversaires_ponderee(
        x_cible, y_cible, teammate_positions, 
        sigma=sigma_optimal  # ← Valeur optimisée par Optuna
    )
    diff_distance = diff_distance_joueurs_proches(
        x_cible, y_cible,
        adv_positions, teammate_positions,
        x_passeur, y_passeur
    )
    # Créer le dictionnaire des features (exactement comme vous l'avez défini)
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
# Import des fonctions depuis Pass_chances_function.py

def create_situation(situation_name):
    """
    Creates realistic game situations (5v5) with professional-level positioning.
    Players are well distributed in width and depth according to tactical principles.
    """
    situations = {
        'aile_droite': {
            'passeur': (0.65, 0.15),  # Ailier droit plus près de la ligne
            'adversaires': np.array([
                [0.78, 0.25],  # Défenseur latéral
                [0.75, 0.40],  # Défenseur central
                [0.60, 0.35],  # Milieu défensif
                [0.70, 0.15],  # Pressing sur le passeur
                [0.50, 0.30]   # Milieu opposé
            ]),
            'coequipiers': np.array([
                [0.40, 0.50],  # Milieu central
                [0.55, 0.25],  # Latéral en soutien
                [0.75, 0.50],  # Attaquant
                [0.85, 0.30],  # Attaquant en profondeur
                [0.60, 0.60]   # Ailier opposé
            ]),
            'title': "Right Wing Attack (5v5)"
        },
        'milieu_terrain': {
            'passeur': (0.50, 0.50),  # Milieu central
            'adversaires': np.array([
                [0.75, 0.30],  # Défenseur central droit
                [0.75, 0.70],  # Défenseur central gauche
                [0.60, 0.40],  # Milieu défensif
                [0.60, 0.60],  # Milieu relayeur
                [0.65, 0.50]   # Attaquant replié
            ]),
            'coequipiers': np.array([
                [0.30, 0.50],  # Milieu défensif
                [0.40, 0.20],  # Latéral droit
                [0.40, 0.80],  # Latéral gauche
                [0.70, 0.35],  # Attaquant droit
                [0.70, 0.65]   # Attaquant gauche
            ]),
            'title': "Midfield Build-up (5v5)"
        },
        'contre_attaque': {
            'passeur': (0.35, 0.50),  # Milieu récupérateur
            'adversaires': np.array([
                [0.45, 0.30],  # Défenseur en repli
                [0.45, 0.70],  # Défenseur en repli
                [0.30, 0.60],  # Milieu en retard
                [0.60, 0.45],  # Défenseur central en repli
                [0.55, 0.55]   # Défenseur en repli déséquilibré
            ]),
            'coequipiers': np.array([
                [0.20, 0.40],  # Soutien défensif
                [0.50, 0.30],  # Ailier droit en course
                [0.55, 0.70],  # Ailier gauche en course 
                [0.65, 0.50],  # Attaquant en appel profondeur
                [0.45, 0.50]   # Soutien offensif
            ]),
            'title': "Fast Counter-Attack (5v5)"
        },
        'surface_reparation': {
            'passeur': (0.85, 0.20),  # Ailier prêt à centrer
            'adversaires': np.array([
                [0.90, 0.25],  # Défenseur proche
                [0.87, 0.45],  # Défenseur central
                [0.92, 0.40],  # Défenseur surface
                [0.80, 0.35],  # Milieu en repli
                [0.85, 0.60]   # Défenseur opposé
            ]),
            'coequipiers': np.array([
                [0.70, 0.15],  # Soutien extérieur
                [0.80, 0.50],  # Attaquant au second poteau
                [0.88, 0.35],  # Attaquant au premier poteau
                [0.75, 0.40],  # Milieu offensif en retrait
                [0.65, 0.45]   # Milieu en soutien
            ]),
            'title': "In the Opponent's Box (5v5)"
        },
        'relance_basse': {
            'passeur': (0.08, 0.50),  # Gardien/défenseur central
            'adversaires': np.array([
                [0.20, 0.30],  # Premier rideau défensif
                [0.20, 0.70],  # Premier rideau défensif
                [0.35, 0.40],  # Second rideau
                [0.35, 0.60],  # Second rideau
                [0.25, 0.50]   # Attaquant en pressing
            ]),
            'coequipiers': np.array([
                [0.15, 0.30],  # Défenseur central proche
                [0.15, 0.70],  # Défenseur central opposé
                [0.25, 0.15],  # Latéral bas
                [0.25, 0.85],  # Latéral haut
                [0.30, 0.50]   # Milieu défensif en soutien
            ]),
            'title': "Build-up from the Back (5v5)"
        },
        'centre_cote': {
            'passeur': (0.95, 0.15),  # Ailier en position de centrer
            'adversaires': np.array([
                [0.90, 0.20],  # Défenseur en couverture directe
                [0.85, 0.30],  # Défenseur central proche
                [0.85, 0.50],  # Défenseur central axial
                [0.88, 0.70],  # Défenseur latéral opposé
                [0.75, 0.40]   # Milieu défensif
            ]),
            'coequipiers': np.array([
                [0.75, 0.20],  # Soutien arrière
                [0.85, 0.40],  # Attaquant premier poteau
                [0.80, 0.60],  # Attaquant second poteau
                [0.70, 0.50],  # Milieu entrée de surface
                [0.90, 0.80]   # Latéral opposé
            ]),
            'title': "Cross from the Wing (5v5)"
        }
    }
    return situations.get(situation_name, situations['milieu_terrain'])

def create_pass_heatmap(origin_x, origin_y, adv_positions, teammate_positions, predictor, 
                       x_divisions=15, y_divisions=10):
    """
    Version mise à jour utilisant les fonctions de Pass_chances_function.py
    """
    # Créer une grille de positions cibles
    x_grid = np.linspace(0, 1, 30)
    y_grid = np.linspace(0, 1, 20)
    
    # Initialiser la matrice de probabilités
    prob_matrix = np.zeros((len(y_grid), len(x_grid)))
    
    # Pour chaque position cible possible
    for i, target_y in enumerate(y_grid):
        for j, target_x in enumerate(x_grid):
            #Construisons les listes de mes deux équipes à retenir
            team_pos = np.array([pos for pos in teammate_positions if distance(pos[0], pos[1], origin_x, origin_y) > 0.005])
            adv_pos = np.array([pos for pos in adv_positions if distance(pos[0], pos[1], origin_x, origin_y) > 0.005])

            # Calculer toutes les features avec les fonctions existantes
            features = calculate_pass_features(
                origin_x, origin_y, target_x, target_y,
                adv_pos, team_pos,
                x_divisions, y_divisions
            )
            
            # Créer un DataFrame avec les features
            feature_df = pd.DataFrame([features])
            
            # Prédire la probabilité
            prob = predictor.predict_proba(feature_df)[0, 1]  # Probabilité classe positive
            prob_matrix[i, j] = prob
    
    return x_grid, y_grid, prob_matrix

def configuration_jeu(idx_frame, csv_home, csv_away):
    df_home = pd.read_csv(csv_home, header=2)
    df_away = pd.read_csv(csv_away, header=2)

    # Filtrer la ligne correspondant à la frame voulue
    home_row = df_home[df_home['Frame'] == idx_frame]
    away_row = df_away[df_away['Frame'] == idx_frame]

    home_pos = []
    away_pos = []

    # Pour chaque joueur home
    if not home_row.empty:
        for col in df_home.columns:
            if col.startswith('Player'):
                col_idx = df_home.columns.get_loc(col)
                # La colonne Y est juste après
                if col_idx + 1 < len(df_home.columns):
                    y_col = df_home.columns[col_idx + 1]
                    x = home_row.iloc[0][col]
                    y = home_row.iloc[0][y_col]
                    if pd.notna(x) and pd.notna(y):
                        home_pos.append((float(x), float(y)))

    # Pour chaque joueur away
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
    Renverse les positions des joueurs sur le terrain selon x et y.
    positions : liste de tuples (x, y)
    pitch_length : longueur du terrain (par défaut 1.0)
    pitch_width  : largeur du terrain (par défaut 1.0)
    Retourne : liste de tuples (x_flipped, y_flipped)
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]

def plot_situation_heatmap_metrica(attacker_pos, defender_pos, ball_pos, frame_id,  predictor, figsize=(12, 8), output_dir=str(HEATMAPS_DIR / "metrica")):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(figsize=figsize)
    create_pitch(ax)
    
    # Extraire les données
    origin_x, origin_y = ball_pos
    adv_positions = np.array(defender_pos)
    teammate_positions = np.array(attacker_pos)
    
    # Afficher les joueurs
    plot_player(ax, origin_x, origin_y, 'home', '10')  # Passeur avec numéro
    
    for i, (x, y) in enumerate(adv_positions):
        plot_player(ax, x, y, 'away')
    
    for i, (x, y) in enumerate(teammate_positions):
        plot_player(ax, x, y, 'home')
    
    # Créer et afficher la carte de chaleur
    x_grid, y_grid, prob_matrix = create_pass_heatmap(
        origin_x, origin_y, adv_positions, teammate_positions, predictor
    )
    
    X, Y = np.meshgrid(x_grid, y_grid)
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=100)
    contour = ax.contourf(X, Y, prob_matrix, levels=50, cmap=cmap, alpha=0.6)
    
    # Ajouter des contours pour plus de clarté
    ax.contour(X, Y, prob_matrix, levels=[0.2, 0.5, 0.8], colors='black', linewidths=0.5, alpha=0.7)
    
    # Légende
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Success Probability', rotation=270, labelpad=20)
    
    save_path = os.path.join(output_dir, f'heatmap_{frame_id}.png')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig, ax

def visualize_game_situation(ax, home_pos, away_pos, ball_pos, goal_threat_value, title=None, show_players=True):
    """
    Visualise une situation de jeu avec les positions des joueurs et la valeur de menace de but.
    
    Parameters:
        ax: Matplotlib axes sur lequel dessiner
        home_pos: Liste de tuples (x, y) pour les positions de l'équipe à domicile
        away_pos: Liste de tuples (x, y) pour les positions de l'équipe à l'extérieur
        ball_pos: Tuple (x, y) pour la position du ballon
        goal_threat_value: Valeur de menace de but pour cette situation
        title: Titre optionnel pour le graphique
        show_players: Booléen indiquant si les joueurs doivent être affichés
    """
    # Créer le terrain
    create_pitch(ax)
    
    # Afficher les joueurs si demandé
    if show_players:
        # Équipe domicile (points bleus)
        x_home = [pos[0] for pos in home_pos]
        y_home = [pos[1] for pos in home_pos]
        ax.scatter(x_home, y_home, color='blue', s=100, zorder=2, label='Home Team', alpha=0.7)
        
        # Équipe extérieur (points rouges)
        x_away = [pos[0] for pos in away_pos]
        y_away = [pos[1] for pos in away_pos]
        ax.scatter(x_away, y_away, color='red', s=100, zorder=2, label='Away Team', alpha=0.7)
    
    # Dessiner le ballon (point jaune plus grand)
    if ball_pos is not None and all(p is not None for p in ball_pos):
        ax.scatter(ball_pos[0], ball_pos[1], color='yellow', s=200, zorder=3, edgecolor='black', label='Ball')
    
    # Ajouter la valeur de menace de but
    if goal_threat_value is not None:
        threat_text = f"Goal Threat: {goal_threat_value:.4f}"
        ax.text(0.05, 0.95, threat_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
    
    # Ajouter un titre si fourni
    if title:
        ax.set_title(title, fontsize=14)
    
    # Ajouter une légende
    ax.legend(loc='lower right', fontsize=10)
    
    return ax


def plot_situation_heatmap(situation_name, predictor, figsize=(12, 8)):
    """
    Affiche une carte de chaleur pour une situation donnée
    """
    # Récupérer la situation
    situation = create_situation(situation_name)
    
    # Configuration
    fig, ax = plt.subplots(figsize=figsize)
    create_pitch(ax)
    
    # Extraire les données
    origin_x, origin_y = situation['passeur']
    adv_positions = situation['adversaires']
    teammate_positions = situation['coequipiers']
    
    # Afficher les joueurs
    plot_player(ax, origin_x, origin_y, 'home', '10')  # Passeur avec numéro
    
    for i, (x, y) in enumerate(adv_positions):
        plot_player(ax, x, y, 'away')
    
    for i, (x, y) in enumerate(teammate_positions):
        plot_player(ax, x, y, 'home')
    
    # Créer et afficher la carte de chaleur
    x_grid, y_grid, prob_matrix = create_pass_heatmap(
        origin_x, origin_y, adv_positions, teammate_positions, predictor
    )
    
    X, Y = np.meshgrid(x_grid, y_grid)
    cmap = LinearSegmentedColormap.from_list('custom', ['red', 'yellow', 'green'], N=100)
    contour = ax.contourf(X, Y, prob_matrix, levels=50, cmap=cmap, alpha=0.6)
    
    # Ajouter des contours pour plus de clarté
    ax.contour(X, Y, prob_matrix, levels=[0.2, 0.5, 0.8], colors='black', linewidths=0.5, alpha=0.7)
    
    # Légende
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Success Probability', rotation=270, labelpad=20)

    plt.tight_layout()
    fig.savefig(str(HEATMAPS_DIR / f'heatmap_{situation_name}.png'), dpi=300)
    return fig, ax

def analyze_pass_features_for_situation(situation_name):
    """
    Analyse les features pour une situation donnée
    """
    situation = create_situation(situation_name)
    origin_x, origin_y = situation['passeur']
    adv_positions = situation['adversaires']
    teammate_positions = situation['coequipiers']
    
    print(f"\n=== Analyse des features pour: {situation['title']} ===")
    print(f"Position du passeur: ({origin_x:.2f}, {origin_y:.2f})")
    print(f"Nombre d'adversaires: {len(adv_positions)}")
    print(f"Nombre de coéquipiers: {len(teammate_positions)}")
    
    # Analyser quelques positions cibles
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

# Fonction utilitaire pour tester une passe spécifique
def test_single_pass(x_passeur, y_passeur, x_cible, y_cible, 
                    adv_positions, teammate_positions, predictor):
    """
    Teste une passe spécifique et retourne la probabilité avec les détails
    """
    features = calculate_pass_features(
        x_passeur, y_passeur, x_cible, y_cible,
        adv_positions, teammate_positions
    )
    
    feature_df = pd.DataFrame([features])
    prob = predictor.predict_proba(feature_df)[0, 1]
    
    return prob, features

# 🤖 CHARGEMENT SÉCURISÉ DU PRÉDICTEUR
def load_predictor_safely():
    """Charge le prédicteur avec fallback"""
    try:
        from pass_predictor import PassPredictor
        predictor = PassPredictor().load_model()
        print("✅ PassPredictor chargé avec succès")
        return predictor, "PassPredictor"
    except Exception as e:
        print(f"⚠️ Erreur PassPredictor: {e}")
        print("💡 Utilisation d'un prédicteur simple")
        

# Charger le prédicteur
predictor, predictor_type = load_predictor_safely()
print(f"🎯 Prédicteur utilisé: {predictor_type}")

# 🧪 TEST DES HYPERPARAMÈTRES
print("🔍 VÉRIFICATION DES HYPERPARAMÈTRES")
print_hyperparameters_info()

# Test d'une prédiction simple
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
    
    # Test de prédiction
    feature_df = pd.DataFrame([features])
    prob = predictor.predict_pass_success(feature_df)[0]
    # Affichage et sauvegarde de toutes les heatmaps pour chaque situation
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
