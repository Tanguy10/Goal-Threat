import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path
import time
# Import the functions from Goal_threat_classes
INTERVAL = 15

from Goal_threat_classes import GoalThreatCalculator, GameState
from config import METRICA_DIR, FIGURES_DIR

def flip_positions(positions: list, pitch_length: float = 1.0, pitch_width: float = 1.0) -> list:
    """
    Flips the players' positions on the pitch along x and y.
    positions : list of (x, y) tuples
    pitch_length : pitch length (default 1.0)
    pitch_width  : pitch width (default 1.0)
    Returns : list of (x_flipped, y_flipped) tuples
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]

import bisect

def _estimate_ball_possession(events_data, frame):
    """
    Estimates ball possession for each frame of `frames`,
    and returns the team in possession.
    """
    # 1) Sort the events once and for all on Start Frame
    events = (
        events_data
        .sort_values('Start Frame')
        .reset_index(drop=True)
    )
    # Extract the start_frames array for bisect
    start_frames = events['Start Frame'].tolist()

    # 2) Find the index of the last event whose start_frame <= f
    idx = bisect.bisect_right(start_frames, frame) - 1

    row = events.iloc[idx]
    team  = row['Team']
    typ   = row['Type']
    # Fix here:
    subtype_val = row['Subtype']
    if pd.isna(subtype_val):
        subtype = ''
    else:
        subtype = str(subtype_val).upper()
    # 3) Apply the business logic to determine who has the ball
    if typ == 'BALL LOST' or (typ == 'CHALLENGE' and subtype.endswith('LOST')):
        # ball lost -> the opponent
        poss = 'Home' if team == 'Away' else 'Away'

    elif typ == 'RECOVERY' or (typ == 'CHALLENGE' and subtype.endswith('WON')):
        # recovery or duel won -> the team in the row
        poss = team

    else:
        # pass, set piece, etc. -> keep the team in the row
        poss = team
    print(f"Possession estimée pour frame {frame}: {poss} (Team: {team}, Type: {typ}, Subtype: {subtype})")
    return poss

import bisect
import pandas as pd

def is_ball_controlled(events_data, current_frame):
    """
    Determines whether the ball is controlled by a player at the given frame
    by analyzing the last recorded event.
    
    Args:
        events_data: DataFrame containing the match events
        current_frame: Number of the frame to check
        
    Returns:
        tuple: (is_controlled, team_possessing)
            - is_controlled (bool): True if the ball is controlled, False otherwise
    """
    # Sort the events on Start Frame (if not already done)
    events = events_data.sort_values('Start Frame').reset_index(drop=True)
    
    # Extract the start_frames array for fast lookup
    start_frames = events['Start Frame'].tolist()
    
    # Find the index of the last event whose start_frame <= current_frame
    idx = bisect.bisect_right(start_frames, current_frame) - 1
    
    if idx < 0:
        return False, None
    
    # Get the last relevant event
    row = events.iloc[idx]
    
    # Check whether the current frame is between the event's start and end
    # If End Frame is NaN or greater than the current frame, the ball is still in play
    if pd.isna(row['End Frame']) or row['End Frame'] > current_frame:   
        return False
    else:
        # The ball is not controlled (between two events, in flight, etc.)
        return True

def _convert_metrica_coordinates(positions, metrica_length=1.0, metrica_width=1.0,
                                target_length=105, target_width=68):
    """
    Converts Metrica coordinates (normalized 0-1) to standard pitch coordinates
    """
    positions_converted = positions.copy()
    
    # Metrica uses normalized coordinates (0,1)
    # Convert to the pitch dimensions
    positions_converted[:, :, 0] *= target_length  # X (length)
    positions_converted[:, :, 1] *= target_width   # Y (width)
    
    return positions_converted

def extract_metrica_data(game_folder, max_frames=500, interval=100, convert_coordinates=False):
    """
    Extracts the Metrica Sports data and prepares it for animation_match
    
    Args:
        game_folder: Path to the folder containing the Metrica data
        max_frames: Maximum number of frames to extract (None = all)
        convert_coordinates: If True, convert coordinates to a standard format
        
    Returns:
        dict with:
        - positions: array (frames, total_players+1, 2) compatible with animation_match
        - ball_carrier_team: array indicating which team has the ball
        - events_data: DataFrame of the events
        - dt: time interval between frames
        - metadata: information about the teams and players
    """
    game_path = Path(game_folder)
    
    # Detect the game number from the folder name
    if "Game_1" in str(game_path):
        game_num = "1"
    elif "Game_2" in str(game_path):
        game_num = "2"
    else:
        # Try to extract it from the folder name
        game_num = "1"  # Default
      # Build the file paths
    events_file = game_path / f"Sample_Game_{game_num}_RawEventsData.csv"
    home_file = game_path / f"Sample_Game_{game_num}_RawTrackingData_Home_Team.csv"
    away_file = game_path / f"Sample_Game_{game_num}_RawTrackingData_Away_Team.csv"
    
    print(f"🔄 Extraction des données Metrica depuis {game_folder}...")
    
    # Check that the files exist
    if not all([events_file.exists(), home_file.exists(), away_file.exists()]):
        print(f"❌ Fichiers manquants dans {game_folder}")
        return None
    
    try:
        # Load the CSV data with the right options
        # Metrica files have 3 header rows, we use the 3rd one
        home_data = pd.read_csv(home_file, header=2)
        away_data = pd.read_csv(away_file, header=2)
        events_data = pd.read_csv(events_file)
        print(f"✅ Données chargées: {len(home_data)} frames")
          # Limit the number of frames if requested
        if max_frames:
            home_data = home_data.head(max_frames)
            away_data = away_data.head(max_frames)
            print(f"📊 Limitation à {max_frames} frames")
        
        home_data = home_data.iloc[::interval].reset_index(drop=True)
        away_data = away_data.iloc[::interval].reset_index(drop=True)
        frames = len(home_data)

        # Identify the player columns (alternating format: PlayerX, Unnamed)
        home_player_ids = []
        away_player_ids = []
        
        # The columns are organized as: Player, Unnamed (Y), Player, Unnamed (Y)...
        for col in home_data.columns:
            if col.startswith('Player'):
                home_player_ids.append(col)
        
        for col in away_data.columns:
            if col.startswith('Player'):
                away_player_ids.append(col)
        
        # Keep the players that have valid data
        home_player_ids = [p for p in home_player_ids if not home_data[p].isna().all()]
        away_player_ids = [p for p in away_player_ids if not away_data[p].isna().all()]
        
        total_players = len(home_player_ids) + len(away_player_ids)
        print(f"👥 Joueurs détectés - Home: {len(home_player_ids)}, Away: {len(away_player_ids)}")
        
        # Initialize the positions array (frames, players+ball, 2)
        positions = np.full((frames, total_players + 1, 2), np.nan)
        ball_carrier_team = np.zeros(frames, dtype=int)  # 0 = Home, 1 = Away
        ball_carrier_liste = []
        # Extract the positions frame by frame
        for frame_idx in range(frames):
            player_idx = 0
            
            # Home players - the Y columns immediately follow the X columns
            for player_id in home_player_ids:
                x_col = player_id
                # Find the Y column (Unnamed that immediately follows)
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
            
            # Away players
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
            
            # Last row for the ball (Metrica uses the last column for the ball)
            # Ball position (last position in the array)
            ball_x, ball_y = np.nan, np.nan

            # Look in home_data
            if 'Ball' in home_data.columns:
                ball_col_idx = home_data.columns.get_loc('Ball')
                ball_y_col_idx = ball_col_idx + 1
                if ball_y_col_idx < len(home_data.columns):
                    ball_y_col = home_data.columns[ball_y_col_idx]
                    ball_x_home = home_data.iloc[frame_idx]['Ball']
                    ball_y_home = home_data.iloc[frame_idx][ball_y_col]
                    if pd.notna(ball_x_home) and pd.notna(ball_y_home):
                        ball_x, ball_y = float(ball_x_home), float(ball_y_home)

            # If the ball position is not valid, look in away_data
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

        # Time interval (Metrica usually uses 25 FPS)
        dt = 0.04  # 1/25 seconds
        
        # Metadata
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
    Plots the evolution of the threat coefficient for each team
    by sampling every `interval` frames.

    data must contain:
      - positions           : np.ndarray de forme (n_frames, n_players+1, 2)
                              (the first n_players rows = players,
                               the last one = ball)
      - ball_carrier_team   : array-like of length n_frames,
                              containing 'home'/'away'/'neutral'
      - dt                  : float, time step between two frames (in s)
    """
    pos         = data['positions']
    carrier     = data['ball_carrier_team']
    dt          = data['dt']

    n_frames, n_entities, _ = pos.shape
    n_players = n_entities - 1   # we consider the last entity to be the ball
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
        # 1) extract player and ball positions
        frame_pos   = pos[frame_idx]          # (n_players+1, 2)
        ball_pos    = frame_pos[-1]           # last = ball
        players_pos = frame_pos[:n_players]   # first = players
        home_players = players_pos[:n_players//2]  # home team players
        away_players = players_pos[n_players//2:n_players]  # away team players
        print(ball_pos)
        if np.isnan(ball_pos).any():
            continue
        # 2) build the lists of home and away player positions
        home_positions = [tuple(home_players[i]) for i in range(len(home_players))]
        away_positions = [tuple(away_players[i]) for i in range(len(away_players))]

        # 3) determine possession
        possession = _estimate_ball_possession(data['events_data'], frame_idx*INTERVAL)
        print(f"Possession pour : {possession}")
        # 4) compute the goal threat
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
        # 5) store
        times.append(frame_idx * dt * INTERVAL)
        threat_h.append(th_h)
        threat_a.append(th_a)

    # final plot
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
    # Convert to array if needed
    h = np.array(threat_h)
    a = np.array(threat_a)

    # Keep only the non-zero values
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
    # Example usage
    game_folder = str(METRICA_DIR / "Game_2_Metrica")

    data = extract_metrica_data(game_folder, max_frames=15000, interval=INTERVAL, convert_coordinates=False)

    plot_goal_threat(data)