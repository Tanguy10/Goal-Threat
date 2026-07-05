import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Goal_threat_classes import GoalThreatCalculator, GameState
from config import METRICA_DIR, PROCESSED_DIR
threat_calculator = GoalThreatCalculator()


def flip_positions(positions: list, pitch_length: float = 1.0, pitch_width: float = 1.0) -> list:
    """
    Flips the players' positions on the pitch along x and y.
    positions : list of (x, y) tuples
    pitch_length : pitch length (default 1.0)
    pitch_width  : pitch width (default 1.0)
    Returns : list of (x_flipped, y_flipped) tuples
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]

def extract_shots_and_preceding_passes(events, n_preceding=2):
    """
    Extracts each SHOT and up to `n_preceding` preceding PASS events.
    
    Parameters:
        events (pd.DataFrame): DataFrame containing your events (with 'Type' and 'Team' columns).
        n_preceding (int): number of preceding events to test (default = 2).
    
    Returns:
        pd.DataFrame: rows corresponding to the shots and their preceding passes (if any),
                      with the 'Team' column preserved.
    """
    # List that will accumulate the Series to concatenate
    rows = []
    # Indices (positions) of all SHOT events
    shot_positions = events.index[events['Type'] == 'SHOT'].tolist()
    
    for pos in shot_positions:
        # For each offset in the window [pos-n_preceding ... pos]
        for offset in range(n_preceding, -1, -1):  # e.g. [2, 1, 0]
            i = pos - offset
            # Check that we stay within the valid range
            if i >= 0 and i < len(events):
                evt = events.iloc[i]
                # Take the event if it is a (preceding) PASS or the SHOT itself
                if offset == 0 or evt['Type'] == 'PASS':
                    rows.append(evt)
                    
    # Rebuild a DataFrame and reset the index
    result = pd.DataFrame(rows).reset_index(drop=True)
    return result

csv_home = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Home_Team.csv")
csv_away = str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawTrackingData_Away_Team.csv")
df_home = pd.read_csv(csv_home, header=2)
df_away = pd.read_csv(csv_away, header=2)

def configuration_jeu(idx_frame, csv_home, csv_away):

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


# Example usage
if __name__ == "__main__":
    df = pd.read_csv(str(METRICA_DIR / "Game_2_Metrica" / "Sample_Game_2_RawEventsData.csv"))
    df_filtered = extract_shots_and_preceding_passes(df)
    print(df_filtered)
    df_filtered.to_csv(str(PROCESSED_DIR / "df_filtered.csv"), index=False)

    liste_goal_threat = [] 
    liste_optimal_paths = []
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
            game_state = GameState.from_positions(ball_pos, home_pos, away_pos)
            value, optimal_path = threat_calculator.calculate_threat(game_state, 2)
            liste_goal_threat.append(value)
            liste_optimal_paths.append(optimal_path)

        if frame.Team == 'Away':
            home_pos, away_pos = configuration_jeu(frame._5, csv_home, csv_away)
            if frame.Period == 2:
                home_pos = flip_positions(home_pos)
                away_pos = flip_positions(away_pos)
                ball_pos = flip_positions([ball_pos])[0]
            print(f"Ball Position: {ball_pos}")
            game_state = GameState.from_positions(ball_pos, away_pos, home_pos)
            value, optimal_path = threat_calculator.calculate_threat(game_state, 2)
            liste_goal_threat.append(value)
            liste_optimal_paths.append(optimal_path)
    print("Liste des valeurs de menace de but :", liste_goal_threat)
    print("Liste des chemins optimaux :", liste_optimal_paths)
    import pickle
    with open(str(PROCESSED_DIR / "liste_goal_threat.pkl"), "wb") as f:
        pickle.dump(liste_goal_threat, f)
    with open(str(PROCESSED_DIR / "liste_optimal_paths.pkl"), "wb") as f:
        pickle.dump(liste_optimal_paths, f)
