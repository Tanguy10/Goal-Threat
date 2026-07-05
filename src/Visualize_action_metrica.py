import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pickle
from config import METRICA_DIR, PROCESSED_DIR, GOAL_THREAT_MAPS_DIR

df_filtered = pd.read_csv(str(PROCESSED_DIR / "df_filtered.csv"))
with open(str(PROCESSED_DIR / "liste_goal_threat.pkl"), "rb") as f:
    liste_goal_threat = pickle.load(f)

csv_home = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Home_Team.csv")
csv_away = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Away_Team.csv")

with open(str(PROCESSED_DIR / "liste_optimal_paths.pkl"), "rb") as f:
    liste_optimal_paths = pickle.load(f)

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

def visualize_game_situation(ax, home_pos, away_pos, ball_pos, goal_threat_value, shot_value, optimal_path = None, title=None, show_players=True):
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
        ax.scatter(ball_pos[0], ball_pos[1], color='yellow', s=50, zorder=3, edgecolor='black', label='Ball')
        # Dessiner le chemin optimal s'il existe
    if optimal_path and len(optimal_path) > 1:
        # Extraire les coordonnées x, y de chaque Zone dans le chemin
        path_x = [zone.x for zone in optimal_path]
        path_y = [zone.y for zone in optimal_path]
        
        # Dessiner des flèches entre chaque point du chemin
        for i in range(len(optimal_path) - 1):
            ax.arrow(path_x[i], path_y[i], 
                    path_x[i+1] - path_x[i], path_y[i+1] - path_y[i],
                    head_width=0.02, head_length=0.03, fc='gold', ec='gold',
                    length_includes_head=True, zorder=4, alpha=0.8,
                    label='Optimal Path' if i == 0 else None)
        
        # Ajouter des points aux positions intermédiaires du chemin
        for i, (x, y) in enumerate(zip(path_x, path_y)):
            if i == 0:  # Position initiale (déjà représentée par le ballon)
                continue
            elif i == len(optimal_path) - 1:  # Position finale
                ax.scatter(x, y, color='gold', s=120, zorder=5, 
                          edgecolor='black', label='Target Position')
            else:  # Positions intermédiaires
                ax.scatter(x, y, color='orange', s=100, zorder=5, 
                          edgecolor='black', label='Pass Receiver' if i == 1 else None)
    # Ajouter la valeur de menace de but
    if goal_threat_value is not None:
        threat_text = f"Goal Threat: {goal_threat_value:.4f}"
        ax.text(0.05, 0.95, threat_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
    if shot_value is not None :
        shot_text = f"Shot Value: {shot_value:.4f}"
        ax.text(0.05, 0.90, shot_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')

    # Ajouter un titre si fourni
    if title:
        ax.set_title(title, fontsize=14)
    
    # Ajouter une légende
    ax.legend(loc='lower left', fontsize=10)
    
    return ax

def save_all_game_situations(df_filtered, liste_goal_threat, liste_optimal_paths, csv_home, csv_away, output_dir=str(GOAL_THREAT_MAPS_DIR / "action_visualizations")):
    """
    Enregistre les visualisations pour toutes les situations de jeu.
    
    Parameters:
        df_filtered: DataFrame contenant les tirs et passes
        liste_goal_threat: Liste des valeurs de menace de but calculées
        csv_home: Chemin vers les données de tracking de l'équipe à domicile
        csv_away: Chemin vers les données de tracking de l'équipe à l'extérieur
        output_dir: Dossier où enregistrer les images
    """
    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Itérer sur chaque ligne, valeur de menace et chemin optimal
    for i, (frame, threat_value, optimal_path) in enumerate(zip(df_filtered.itertuples(), liste_goal_threat, liste_optimal_paths)):
        # Créer une nouvelle figure pour chaque situation
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Obtenir la position du ballon
        ball_pos = (frame._11, frame._12)  # Maintenant que nous connaissons les noms des colonnes
        
        # Obtenir les positions des joueurs
        if frame.Team == 'Home':
            home_pos, away_pos = configuration_jeu(frame._5, csv_home, csv_away)
            if frame.Period == 1:
                home_pos = flip_positions(home_pos)
                away_pos = flip_positions(away_pos)
                ball_pos = flip_positions([ball_pos])[0]
        else:  # 'Away'
            home_pos, away_pos = configuration_jeu(frame._5, csv_home, csv_away)
            if frame.Period == 2:
                home_pos = flip_positions(home_pos)
                away_pos = flip_positions(away_pos)
                ball_pos = flip_positions([ball_pos])[0]
        
        # Créer un titre informatif
        event_type = "Shot" if frame.Type == 'SHOT' else "Pass"
        passes_info = f" ({len(optimal_path)-1} passes)" if optimal_path and len(optimal_path) > 1 else ""
        title = f"Event {frame._5}: {event_type}{passes_info}, Threat: {threat_value:.4f}"
        
        # Visualiser la situation avec le chemin optimal
        visualize_game_situation(ax, home_pos, away_pos, ball_pos, threat_value, shot_value=None, optimal_path=optimal_path, title=title)

        # Enregistrer l'image
        filename = f"{output_dir}/situation_{i+1:02d}_{frame.Team}_{event_type}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image enregistrée: {filename}")

save_all_game_situations(df_filtered, liste_goal_threat, liste_optimal_paths, csv_home, csv_away)
