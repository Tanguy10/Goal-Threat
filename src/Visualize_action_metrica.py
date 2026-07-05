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

def flip_positions(positions: list, pitch_length: float = 1.0, pitch_width: float = 1.0) -> list:
    """
    Flips the players' positions on the pitch along x and y.
    positions : list of (x, y) tuples
    pitch_length : pitch length (default 1.0)
    pitch_width  : pitch width (default 1.0)
    Returns : list of (x_flipped, y_flipped) tuples
    """
    return [(pitch_length - x, pitch_width - y) for x, y in positions]


def create_pitch(ax):
    """Create a standardized football pitch (normalized proportions)"""
    # Pitch (green rectangle)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                           edgecolor='white', facecolor='#538032', alpha=0.8)
    ax.add_patch(rect)
    
    # Halfway line
    plt.plot([0.5, 0.5], [0, 1], color='white', linewidth=2)
    
    # Center circle
    circle = plt.Circle((0.5, 0.5), 0.1, color='white', fill=False, linewidth=2)
    ax.add_patch(circle)
    
    # Penalty area (left)
    rect = patches.Rectangle((0, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Goal area (left)
    rect = patches.Rectangle((0, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Penalty area (right)
    rect = patches.Rectangle((0.82, 0.3), 0.18, 0.4, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Goal area (right)
    rect = patches.Rectangle((0.94, 0.4), 0.06, 0.2, linewidth=2, 
                           edgecolor='white', facecolor='none')
    ax.add_patch(rect)
    
    # Adjust the plot parameters
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def visualize_game_situation(ax, home_pos, away_pos, ball_pos, goal_threat_value, shot_value, optimal_path = None, title=None, show_players=True):
    """
    Visualizes a game situation with the players' positions and the goal threat value.
    
    Parameters:
        ax: Matplotlib axes to draw on
        home_pos: List of (x, y) tuples for the home team positions
        away_pos: List of (x, y) tuples for the away team positions
        ball_pos: Tuple (x, y) for the ball position
        goal_threat_value: Goal threat value for this situation
        title: Optional title for the plot
        show_players: Boolean indicating whether the players should be shown
    """
    # Create the pitch
    create_pitch(ax)
    
    # Show the players if requested
    if show_players:
        # Home team (blue dots)
        x_home = [pos[0] for pos in home_pos]
        y_home = [pos[1] for pos in home_pos]
        ax.scatter(x_home, y_home, color='blue', s=100, zorder=2, label='Home Team', alpha=0.7)
        
        # Away team (red dots)
        x_away = [pos[0] for pos in away_pos]
        y_away = [pos[1] for pos in away_pos]
        ax.scatter(x_away, y_away, color='red', s=100, zorder=2, label='Away Team', alpha=0.7)
    
    # Draw the ball (bigger yellow dot)
    if ball_pos is not None and all(p is not None for p in ball_pos):
        ax.scatter(ball_pos[0], ball_pos[1], color='yellow', s=50, zorder=3, edgecolor='black', label='Ball')
        # Draw the optimal path if it exists
    if optimal_path and len(optimal_path) > 1:
        # Extract the x, y coordinates of each Zone in the path
        path_x = [zone.x for zone in optimal_path]
        path_y = [zone.y for zone in optimal_path]
        
        # Draw arrows between each point of the path
        for i in range(len(optimal_path) - 1):
            ax.arrow(path_x[i], path_y[i], 
                    path_x[i+1] - path_x[i], path_y[i+1] - path_y[i],
                    head_width=0.02, head_length=0.03, fc='gold', ec='gold',
                    length_includes_head=True, zorder=4, alpha=0.8,
                    label='Optimal Path' if i == 0 else None)
        
        # Add points at the intermediate positions of the path
        for i, (x, y) in enumerate(zip(path_x, path_y)):
            if i == 0:  # Initial position (already shown by the ball)
                continue
            elif i == len(optimal_path) - 1:  # Final position
                ax.scatter(x, y, color='gold', s=120, zorder=5, 
                          edgecolor='black', label='Target Position')
            else:  # Intermediate positions
                ax.scatter(x, y, color='orange', s=100, zorder=5, 
                          edgecolor='black', label='Pass Receiver' if i == 1 else None)
    # Add the goal threat value
    if goal_threat_value is not None:
        threat_text = f"Goal Threat: {goal_threat_value:.4f}"
        ax.text(0.05, 0.95, threat_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
    if shot_value is not None :
        shot_text = f"Shot Value: {shot_value:.4f}"
        ax.text(0.05, 0.90, shot_text, transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')

    # Add a title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add a legend
    ax.legend(loc='lower left', fontsize=10)
    
    return ax

def save_all_game_situations(df_filtered, liste_goal_threat, liste_optimal_paths, csv_home, csv_away, output_dir=str(GOAL_THREAT_MAPS_DIR / "action_visualizations")):
    """
    Saves the visualizations for all game situations.
    
    Parameters:
        df_filtered: DataFrame containing the shots and passes
        liste_goal_threat: List of the computed goal threat values
        csv_home: Path to the home team tracking data
        csv_away: Path to the away team tracking data
        output_dir: Folder where the images are saved
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over each row, threat value and optimal path
    for i, (frame, threat_value, optimal_path) in enumerate(zip(df_filtered.itertuples(), liste_goal_threat, liste_optimal_paths)):
        # Create a new figure for each situation
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Get the ball position
        ball_pos = (frame._11, frame._12)  # Now that we know the column names
        
        # Get the players' positions
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
        
        # Create an informative title
        event_type = "Shot" if frame.Type == 'SHOT' else "Pass"
        passes_info = f" ({len(optimal_path)-1} passes)" if optimal_path and len(optimal_path) > 1 else ""
        title = f"Event {frame._5}: {event_type}{passes_info}, Threat: {threat_value:.4f}"
        
        # Visualize the situation with the optimal path
        visualize_game_situation(ax, home_pos, away_pos, ball_pos, threat_value, shot_value=None, optimal_path=optimal_path, title=title)

        # Save the image
        filename = f"{output_dir}/situation_{i+1:02d}_{frame.Team}_{event_type}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image enregistrée: {filename}")

save_all_game_situations(df_filtered, liste_goal_threat, liste_optimal_paths, csv_home, csv_away)
