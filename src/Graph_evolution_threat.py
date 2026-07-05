import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path
import time
# Importer les fonctions de Goal_threat_classes
INTERVAL = 15

from Goal_threat_classes import GoalThreatCalculator, GameState
from config import METRICA_DIR, FIGURES_DIR

def flip_positions(positions: list, pitch_length: float = 1.0, pitch_width: float = 1.0) -> list:
    """
    Renverse les positions des joueurs sur le terrain selon x et y.
    positions : liste de tuples (x, y)
    pitch_length : longueur du terrain (par défaut 1.0)
    pitch_width  : largeur du terrain (par défaut 1.0)
    Retourne : liste de tuples (x_flipped, y_flipped)
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]

import bisect

def _estimate_ball_possession(events_data, frame):
    """
    Estime la possession du ballon pour chaque frame de `frames`,
    et renvoie l'équipe en possession.
    """
    # 1) Trier une bonne fois pour toutes les événements sur Start Frame
    events = (
        events_data
        .sort_values('Start Frame')
        .reset_index(drop=True)
    )
    # Extraire le tableau de start_frames pour bisect
    start_frames = events['Start Frame'].tolist()

    # 2) Trouver l'index du dernier événement dont start_frame <= f
    idx = bisect.bisect_right(start_frames, frame) - 1

    row = events.iloc[idx]
    team  = row['Team']
    typ   = row['Type']
    # Correction ici :
    subtype_val = row['Subtype']
    if pd.isna(subtype_val):
        subtype = ''
    else:
        subtype = str(subtype_val).upper()
    # 3) Appliquer votre logique métier pour déterminer qui a la balle
    if typ == 'BALL LOST' or (typ == 'CHALLENGE' and subtype.endswith('LOST')):
        # perte de balle → l’adversaire
        poss = 'Home' if team == 'Away' else 'Away'

    elif typ == 'RECOVERY' or (typ == 'CHALLENGE' and subtype.endswith('WON')):
        # récupération ou duel gagné → l'équipe du row
        poss = team

    else:
        # passe, set piece, etc. → on reste avec l'équipe du row
        poss = team
    print(f"Possession estimée pour frame {frame}: {poss} (Team: {team}, Type: {typ}, Subtype: {subtype})")
    return poss

import bisect
import pandas as pd

def is_ball_controlled(events_data, current_frame):
    """
    Détermine si le ballon est contrôlé par un joueur à la frame spécifiée
    en analysant le dernier événement enregistré.
    
    Args:
        events_data: DataFrame contenant les événements du match
        current_frame: Numéro de la frame à vérifier
        
    Returns:
        tuple: (is_controlled, team_possessing)
            - is_controlled (bool): True si le ballon est contrôlé, False sinon
    """
    # Trier les événements sur Start Frame (si pas déjà fait)
    events = events_data.sort_values('Start Frame').reset_index(drop=True)
    
    # Extraire le tableau de start_frames pour recherche rapide
    start_frames = events['Start Frame'].tolist()
    
    # Trouver l'index du dernier événement dont start_frame <= current_frame
    idx = bisect.bisect_right(start_frames, current_frame) - 1
    
    if idx < 0:
        return False, None
    
    # Récupérer le dernier événement pertinent
    row = events.iloc[idx]
    
    # Vérifier si la frame actuelle se trouve entre le début et la fin de l'événement
    # Si End Frame est NaN ou supérieur à la frame actuelle, le ballon est toujours en jeu
    if pd.isna(row['End Frame']) or row['End Frame'] > current_frame:   
        return False
    else:
        # Le ballon n'est pas contrôlé (entre deux événements, en vol, etc.)
        return True

def _convert_metrica_coordinates(positions, metrica_length=1.0, metrica_width=1.0,
                                target_length=105, target_width=68):
    """
    Convertit les coordonnées Metrica (normalisées 0-1) vers des coordonnées de terrain standard
    """
    positions_converted = positions.copy()
    
    # Metrica utilise des coordonnées normalisées (0,1)
    # Convertir vers les dimensions du terrain
    positions_converted[:, :, 0] *= target_length  # X (longueur)
    positions_converted[:, :, 1] *= target_width   # Y (largeur)
    
    return positions_converted

def extract_metrica_data(game_folder, max_frames=500, interval=100, convert_coordinates=False):
    """
    Extrait les données Metrica Sports et les prépare pour animation_match
    
    Args:
        game_folder: Chemin vers le dossier contenant les données Metrica
        max_frames: Nombre maximum de frames à extraire (None = toutes)
        convert_coordinates: Si True, convertit les coordonnées vers un format standard
        
    Returns:
        dict avec:
        - positions: array (frames, total_players+1, 2) compatible avec animation_match
        - ball_carrier_team: array indiquant quelle équipe a le ballon
        - events_data: DataFrame des événements
        - dt: intervalle de temps entre frames
        - metadata: informations sur les équipes et joueurs
    """
    game_path = Path(game_folder)
    
    # Détecter le numéro du jeu depuis le nom du dossier
    if "Game_1" in str(game_path):
        game_num = "1"
    elif "Game_2" in str(game_path):
        game_num = "2"
    else:
        # Essayer d'extraire depuis le nom du dossier
        game_num = "1"  # Par défaut
      # Construire les chemins des fichiers
    events_file = game_path / f"Sample_Game_{game_num}_RawEventsData.csv"
    home_file = game_path / f"Sample_Game_{game_num}_RawTrackingData_Home_Team.csv"
    away_file = game_path / f"Sample_Game_{game_num}_RawTrackingData_Away_Team.csv"
    
    print(f"🔄 Extraction des données Metrica depuis {game_folder}...")
    
    # Vérifier l'existence des fichiers
    if not all([events_file.exists(), home_file.exists(), away_file.exists()]):
        print(f"❌ Fichiers manquants dans {game_folder}")
        return None
    
    try:
        # Charger les données CSV avec les bonnes options
        # Les fichiers Metrica ont 3 lignes d'en-têtes, on utilise la 3ème
        home_data = pd.read_csv(home_file, header=2)
        away_data = pd.read_csv(away_file, header=2)
        events_data = pd.read_csv(events_file)
        print(f"✅ Données chargées: {len(home_data)} frames")
          # Limiter le nombre de frames si demandé
        if max_frames:
            home_data = home_data.head(max_frames)
            away_data = away_data.head(max_frames)
            print(f"📊 Limitation à {max_frames} frames")
        
        home_data = home_data.iloc[::interval].reset_index(drop=True)
        away_data = away_data.iloc[::interval].reset_index(drop=True)
        frames = len(home_data)

        # Identifier les colonnes de joueurs (format alterné: PlayerX, Unnamed)
        home_player_ids = []
        away_player_ids = []
        
        # Les colonnes sont organisées: Player, Unnamed (Y), Player, Unnamed (Y)...
        for col in home_data.columns:
            if col.startswith('Player'):
                home_player_ids.append(col)
        
        for col in away_data.columns:
            if col.startswith('Player'):
                away_player_ids.append(col)
        
        # Filtrer les joueurs qui ont des données valides
        home_player_ids = [p for p in home_player_ids if not home_data[p].isna().all()]
        away_player_ids = [p for p in away_player_ids if not away_data[p].isna().all()]
        
        total_players = len(home_player_ids) + len(away_player_ids)
        print(f"👥 Joueurs détectés - Home: {len(home_player_ids)}, Away: {len(away_player_ids)}")
        
        # Initialiser le tableau des positions (frames, joueurs+ballon, 2)
        positions = np.full((frames, total_players + 1, 2), np.nan)
        ball_carrier_team = np.zeros(frames, dtype=int)  # 0 = Home, 1 = Away
        ball_carrier_liste = []
        # Extraire les positions frame par frame
        for frame_idx in range(frames):
            player_idx = 0
            
            # Joueurs Home - Les colonnes Y suivent immédiatement les colonnes X
            for player_id in home_player_ids:
                x_col = player_id
                # Trouver la colonne Y (Unnamed qui suit immédiatement)
                x_col_idx = home_data.columns.get_loc(x_col)
                y_col_idx = x_col_idx + 1
                
                if y_col_idx < len(home_data.columns):
                    y_col = home_data.columns[y_col_idx]
                    
                    x_val = home_data.iloc[frame_idx][x_col]
                    y_val = home_data.iloc[frame_idx][y_col]
                    
                    if pd.notna(x_val) and pd.notna(y_val):
                        positions[frame_idx, player_idx, 0] = float(x_val)
                        positions[frame_idx, player_idx, 1] = float(y_val)
                
                player_idx += 1
            
            # Joueurs Away
            for player_id in away_player_ids:
                x_col = player_id
                x_col_idx = away_data.columns.get_loc(x_col)
                y_col_idx = x_col_idx + 1
                
                if y_col_idx < len(away_data.columns):
                    y_col = away_data.columns[y_col_idx]
                    
                    x_val = away_data.iloc[frame_idx][x_col]
                    y_val = away_data.iloc[frame_idx][y_col]
                    
                    if pd.notna(x_val) and pd.notna(y_val):
                        positions[frame_idx, player_idx, 0] = float(x_val)
                        positions[frame_idx, player_idx, 1] = float(y_val)
                
                player_idx += 1
            
            # Dernière ligne pour le ballon (Metrica utilise la dernière colonne pour le ballon)
            # Position du ballon (dernière position dans le tableau)
            ball_x, ball_y = np.nan, np.nan

            # Chercher dans home_data
            if 'Ball' in home_data.columns:
                ball_col_idx = home_data.columns.get_loc('Ball')
                ball_y_col_idx = ball_col_idx + 1
                if ball_y_col_idx < len(home_data.columns):
                    ball_y_col = home_data.columns[ball_y_col_idx]
                    ball_x_home = home_data.iloc[frame_idx]['Ball']
                    ball_y_home = home_data.iloc[frame_idx][ball_y_col]
                    if pd.notna(ball_x_home) and pd.notna(ball_y_home):
                        ball_x, ball_y = float(ball_x_home), float(ball_y_home)

            # Si la position du ballon n'est pas valide, chercher dans away_data
            if (not pd.notna(ball_x)) or (not pd.notna(ball_y)):
                if 'Ball' in away_data.columns:
                    ball_col_idx = away_data.columns.get_loc('Ball')
                    ball_y_col_idx = ball_col_idx + 1
                    if ball_y_col_idx < len(away_data.columns):
                        ball_y_col = away_data.columns[ball_y_col_idx]
                        ball_x_away = away_data.iloc[frame_idx]['Ball']
                        ball_y_away = away_data.iloc[frame_idx][ball_y_col]
                        if pd.notna(ball_x_away) and pd.notna(ball_y_away):
                            ball_x, ball_y = float(ball_x_away), float(ball_y_away)
            ball_carrier_liste.append(_estimate_ball_possession(events_data, frame_idx))
            print(f"{ball_carrier_team[frame_idx]} a la balle")

            positions[frame_idx, -1, 0] = ball_x
            positions[frame_idx, -1, 1] = ball_y
            print(f"Coordonnées de la balle chargées : ({positions[frame_idx, -1, 0]}, {positions[frame_idx, -1, 1]})")

        # Intervalle de temps (Metrica utilise généralement 25 FPS)
        dt = 0.04  # 1/25 secondes
        
        # Métadonnées
        metadata = {
            'game_number': game_num,
            'total_frames': frames,
            'home_players': len(home_player_ids),
            'away_players': len(away_player_ids),
            'fps': 25,
            'dt': dt
        }
        
        print(f"✅ Extraction terminée: {frames} frames, {total_players} joueurs")
        
        return {
            'positions': positions,
            'ball_carrier_team': ball_carrier_team,
            'events_data': events_data,
            'dt': dt,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_goal_threat(data: dict,
                     interval: int = 1):
    """
    Trace l'évolution du coefficient de menace pour chaque équipe
    en échantillonnant toutes les `interval` frames.

    data doit contenir :
      - positions           : np.ndarray de forme (n_frames, n_players+1, 2)
                              (les n_players premières lignes = joueurs,
                               la dernière = ballon)
      - ball_carrier_team   : array-like de longueur n_frames,
                              contenant 'home'/'away'/'neutral'
      - dt                  : float, pas de temps entre deux frames (en s)
    """
    pos         = data['positions']
    carrier     = data['ball_carrier_team']
    dt          = data['dt']

    n_frames, n_entities, _ = pos.shape
    n_players = n_entities - 1   # on considère que la dernière entité = ballon
    threat_calculator = GoalThreatCalculator()
    times      = []
    threat_h   = []
    threat_a   = []
    start = time.time()
    print(n_frames)
    for frame_idx in range(0, n_frames, interval):
        is_controlled = is_ball_controlled(data['events_data'], frame_idx * INTERVAL)
        if not is_controlled:
            print(f"Frame {frame_idx} : ballon non contrôlé, skip")
            continue
        print(f"Frame {frame_idx} : ballon contrôlé, calcul de la menace")
        print(f"Temps de calcul: {time.time() - start:.2f}s")
        # 1) extraire positions joueurs et ballon
        frame_pos   = pos[frame_idx]          # (n_players+1, 2)
        ball_pos    = frame_pos[-1]           # dernier = ballon
        players_pos = frame_pos[:n_players]   # premiers = joueurs
        home_players = players_pos[:n_players//2]  # joueurs de l'équipe home
        away_players = players_pos[n_players//2:n_players]  # joueurs de l'équipe away
        print(ball_pos)
        if np.isnan(ball_pos).any():
            continue
        # 2) construire les listes des positions des joueurs home et away
        home_positions = [tuple(home_players[i]) for i in range(len(home_players))]
        away_positions = [tuple(away_players[i]) for i in range(len(away_players))]

        # 3) déterminer la possession
        possession = _estimate_ball_possession(data['events_data'], frame_idx*INTERVAL)
        print(f"Possession pour : {possession}")
        # 4) calcul du goal threat
        if possession == 'Home':
            away_positions = flip_positions(away_positions)
            home_positions = flip_positions(home_positions)
            ball_pos = flip_positions([ball_pos])[0]
            game_state = GameState.from_positions(ball_pos, home_positions, away_positions)
            th_h = threat_calculator.calculate_threat(game_state, 2)[0]
            th_a = 0   
        else :
            th_h = 0
            game_state = GameState.from_positions(ball_pos, away_positions, home_positions)
            th_a = threat_calculator.calculate_threat(game_state, 2)[0]
        # 5) stocker
        times.append(frame_idx * dt * INTERVAL)
        threat_h.append(th_h)
        threat_a.append(th_a)

    # tracé final
    plt.figure(figsize=(10, 4))
    plt.plot(times, threat_h, label='Home Threat', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Goal Threat Home')
    plt.title('Evolution of Goal Threat Home')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "goal_threat_home.png"))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(times, threat_a, label='Away Threat', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Goal Threat Away')
    plt.title('Evolution of Goal Threat Away')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "goal_threat_away.png"))
    plt.close()

    print("Statistiques de menace calculées (sans enlever les 0):")
    mean_home = np.mean(threat_h)
    mean_away = np.mean(threat_a)
    print(f"Menace moyenne Home: {mean_home:.2f}, Menace moyenne Away: {mean_away:.2f}")
    variance_home = np.var(threat_h)
    variance_away = np.var(threat_a)
    print(f"Variance Home: {variance_home:.2f}, Variance Away: {variance_away:.2f}")
    quantile_95_home = np.quantile(threat_h, 0.95)
    quantile_95_away = np.quantile(threat_a, 0.95)
    print(f"Quantile 95 Home: {quantile_95_home:.2f}, Quantile 95 Away: {quantile_95_away:.2f}")
    percentile_home = np.percentile(threat_h, [0, 25, 50, 75, 100])
    percentile_away = np.percentile(threat_a, [0, 25, 50, 75, 100])
    print("Statistiques calculées :")
    print(f"  - Home: Moyenne={mean_home:.2f}, Variance={variance_home:.2f}, Quantile 95={quantile_95_home:.2f}, Percentiles={percentile_home}")
    print(f"  - Away: Moyenne={mean_away:.2f}, Variance={variance_away:.2f}, Quantile 95={quantile_95_away:.2f}, Percentiles={percentile_away}")

    print("Statistiques de menace calculées (en enlevant les 0):")
    # Convertir en array si besoin
    h = np.array(threat_h)
    a = np.array(threat_a)

    # Ne garder que les valeurs non nulles
    h_nz = h[h != 0]
    a_nz = a[a != 0]
    mean_home = np.mean(threat_h)
    mean_away = np.mean(threat_a)
    print(f"Menace moyenne Home: {mean_home:.2f}, Menace moyenne Away: {mean_away:.2f}")
    variance_home = np.var(threat_h)
    variance_away = np.var(threat_a)
    print(f"Variance Home: {variance_home:.2f}, Variance Away: {variance_away:.2f}")
    quantile_95_home = np.quantile(threat_h, 0.95)
    quantile_95_away = np.quantile(threat_a, 0.95)
    print(f"Quantile 95 Home: {quantile_95_home:.2f}, Quantile 95 Away: {quantile_95_away:.2f}")
    percentile_home = np.percentile(threat_h, [0, 25, 50, 75, 100])
    percentile_away = np.percentile(threat_a, [0, 25, 50, 75, 100])
    print("Statistiques calculées :")
    print(f"  - Home: Moyenne={mean_home:.2f}, Variance={variance_home:.2f}, Quantile 95={quantile_95_home:.2f}, Percentiles={percentile_home}")
    print(f"  - Away: Moyenne={mean_away:.2f}, Variance={variance_away:.2f}, Quantile 95={quantile_95_away:.2f}, Percentiles={percentile_away}")



if __name__ == "__main__":
    # Exemple d'utilisation
    game_folder = str(METRICA_DIR / "Game_2_Metrica")

    data = extract_metrica_data(game_folder, max_frames=15000, interval=INTERVAL, convert_coordinates=False)

    plot_goal_threat(data)