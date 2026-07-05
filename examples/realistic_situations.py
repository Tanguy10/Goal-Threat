"""
Realistic tactical situations to test the goal threat function
Based on authentic formations and game scenarios
"""

def situation_contre_attaque_rapide():
    """
    Classic 3v2 counter-attack
    - Attacking team: fast transition with 3 advanced players
    - Defending team: only 2 center-backs repositioned
    - Ball carrier in midfield with 2 wingers in support
    """
    # Attacking formation: 4-3-3 in offensive transition
    teammates_position = [
        # Goalkeeper
        (0.05, 0.5),
        # Defenders who stayed back  
        (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8),
        # Midfielders pushing up
        (0.45, 0.35), (0.45, 0.65),
        # Forwards on the counter
        (0.75, 0.3),   # Left winger
        (0.8, 0.5),    # Center-forward (ball carrier)
        (0.75, 0.7),   # Right winger
        (0.7, 0.5)     # Attacking midfielder in support
    ]
    
    # Unbalanced defense - only a few players recovered
    defenders_position = [
        # Opposing goalkeeper
        (0.95, 0.5),
        # Center-backs
        (0.85, 0.45), (0.85, 0.55),
        # Fullbacks not yet recovered
        (0.6, 0.15), (0.6, 0.85),
        # Defensive midfielders
        (0.7, 0.4), (0.7, 0.6)
    ]
    
    ball_zone = (0.8, 0.5)  # Ball with the center-forward
    return teammates_position, defenders_position, ball_zone

def situation_action_aile_centrage():
    """
    Action on the right wing with an imminent cross
    - Right winger in a crossing position
    - Forwards in finishing positions in the box
    - Organized defense but under pressure
    """
    teammates_position = [
        # Goalkeeper and defense
        (0.05, 0.5),
        (0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
        # Midfielders
        (0.4, 0.4), (0.5, 0.5), (0.4, 0.6),
        # Forwards in the box
        (0.88, 0.45),  # Center-forward
        (0.85, 0.6),   # Second forward
        (0.92, 0.15)   # Right winger (ball carrier)
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Central defense
        (0.9, 0.4), (0.9, 0.6),
        # Fullbacks
        (0.85, 0.2), (0.88, 0.8),
        # Defensive midfielders
        (0.75, 0.35), (0.75, 0.55), (0.8, 0.45)
    ]
    
    ball_zone = (0.92, 0.15)  # Winger in a crossing position
    return teammates_position, defenders_position, ball_zone

def situation_coup_franc_dangereux():
    """
    Free kick 25m from goal
    - Direct kicker facing the goal
    - Organized defensive wall
    - Forwards positioned for a rebound or deflection
    """
    teammates_position = [
        # Goalkeeper and defenders
        (0.05, 0.5),
        (0.3, 0.3), (0.3, 0.5), (0.3, 0.7),
        # Midfielders
        (0.5, 0.4), (0.5, 0.6),
        # Players in the box for a rebound
        (0.88, 0.4), (0.88, 0.6), (0.9, 0.5),
        # Kicker
        (0.75, 0.5),
        # Player for a short pass
        (0.72, 0.4)
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Defensive wall
        (0.82, 0.45), (0.82, 0.48), (0.82, 0.52), (0.82, 0.55),
        # Defenders marking in the box
        (0.9, 0.4), (0.9, 0.6), (0.93, 0.5),
        # Defensive midfielders
        (0.75, 0.3), (0.75, 0.7)
    ]
    
    ball_zone = (0.75, 0.5)  # Free kick position
    return teammates_position, defenders_position, ball_zone

def situation_corner_offensif():
    """
    Attacking corner with realistic tactical organization
    - Corner taker
    - Forwards positioned in the finishing zones
    - Defense in mixed marking
    """
    teammates_position = [
        # Goalkeeper and defender who stayed back
        (0.05, 0.5), (0.5, 0.5),
        # Midfielders pushed up
        (0.75, 0.3), (0.8, 0.4),
        # Forwards in the box
        (0.92, 0.45),  # Near post
        (0.88, 0.5),   # Center of the box
        (0.85, 0.55),  # Far post
        (0.82, 0.4),   # Penalty spot
        # Players for the counter
        (0.7, 0.6), (0.65, 0.35),
        # Corner taker
        (1.0, 0.05)
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Defenders on the line
        (0.98, 0.45), (0.98, 0.55),
        # Marking in the box
        (0.9, 0.45), (0.86, 0.5), (0.83, 0.55),
        # Defenders marking
        (0.82, 0.4), (0.88, 0.4),
        # Player on the back post
        (0.99, 0.3),
        # Defensive midfielder for clearance
        (0.75, 0.5)
    ]
    
    ball_zone = (1.0, 0.05)  # Right corner
    return teammates_position, defenders_position, ball_zone

def situation_rupture_milieu():
    """
    Break through the midfield
    - Attacking midfielder carrying the ball
    - Central drive between the lines
    - Forwards moving for a combination
    """
    teammates_position = [
        # Goalkeeper and defense
        (0.05, 0.5),
        (0.2, 0.25), (0.2, 0.4), (0.2, 0.6), (0.2, 0.75),
        # Midfielders
        (0.4, 0.3), (0.4, 0.7),  # Wide midfielders
        # Break and attack
        (0.65, 0.5),   # Attacking midfielder (ball carrier)
        (0.8, 0.4),    # Forward making a run
        (0.8, 0.6),    # Second forward
        (0.75, 0.5)    # Playmaker in support
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Organized defense
        (0.85, 0.3), (0.85, 0.45), (0.85, 0.55), (0.85, 0.7),
        # Defensive midfielders
        (0.7, 0.4), (0.7, 0.6),
        # Ball-winning midfielder
        (0.6, 0.5)
    ]
    
    ball_zone = (0.65, 0.5)  # Attacking midfielder breaking through
    return teammates_position, defenders_position, ball_zone

def situation_une_deux_surface():
    """
    One-two at the edge of the penalty area
    - Quick combination between two forwards
    - Compact defense but caught out
    - Likely finish
    """
    teammates_position = [
        # Goalkeeper and defense
        (0.05, 0.5),
        (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8),
        # Midfielders in support
        (0.45, 0.4), (0.45, 0.6), (0.5, 0.5),
        # Combination at the edge of the box
        (0.83, 0.5),   # First forward (ball carrier)
        (0.87, 0.45),  # Second forward for the lay-off
        (0.8, 0.6)     # Third man
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Tight defense
        (0.88, 0.4), (0.88, 0.6),
        (0.85, 0.45), (0.85, 0.55),
        # Defensive midfielders
        (0.75, 0.35), (0.75, 0.65),
        (0.8, 0.5)
    ]
    
    ball_zone = (0.83, 0.5)  # At the edge of the box
    return teammates_position, defenders_position, ball_zone

def situation_pressing_haut():
    """
    High press with ball recovery
    - Team that has just won the ball high
    - Opponent's defense off-balance
    - Immediate attacking transition
    """
    teammates_position = [
        # Goalkeeper and high defense
        (0.05, 0.5),
        (0.45, 0.2), (0.45, 0.4), (0.45, 0.6), (0.45, 0.8),
        # High midfielders
        (0.6, 0.35), (0.6, 0.65),
        # Press and transition
        (0.75, 0.5),   # Ball-winner (ball carrier)
        (0.8, 0.4),    # Nearby forward
        (0.8, 0.6),    # Second forward
        (0.72, 0.5)    # Playmaker in support
    ]
    
    defenders_position = [
        # Goalkeeper
        (0.95, 0.5),
        # Surprised, high defense
        (0.7, 0.3), (0.7, 0.7),
        # Center-backs not yet repositioned
        (0.82, 0.45), (0.82, 0.55),
        # Midfielders recovering
        (0.68, 0.4), (0.68, 0.6)
    ]
    
    ball_zone = (0.75, 0.5)  # Ball won high
    return teammates_position, defenders_position, ball_zone

def situation_sortie_balle_arretee():
    """
    Build-up from a set piece with attacking organization
    - Restart from a defensive free kick
    - Organized transition to attack
    - Defense recovering position
    """
    teammates_position = [
        # Goalkeeper
        (0.05, 0.5),
        # Spread defense for the build-up
        (0.2, 0.2), (0.2, 0.8), (0.15, 0.4), (0.15, 0.6),
        # Midfielders in support
        (0.35, 0.3), (0.35, 0.7), (0.4, 0.5),
        # Forwards on the move
        (0.65, 0.4), (0.65, 0.6),
        # Player starting the build-up
        (0.25, 0.5)
    ]
    
    defenders_position = [
        # Opposing goalkeeper
        (0.95, 0.5),
        # Defense recovering position
        (0.8, 0.25), (0.8, 0.45), (0.8, 0.55), (0.8, 0.75),
        # Midfielders recovering
        (0.6, 0.4), (0.6, 0.6), (0.7, 0.5)
    ]
    
    ball_zone = (0.25, 0.5)  # Ball with the build-up player
    return teammates_position, defenders_position, ball_zone

def situation_attaque_organisee_vers_la_droite():
    """
    Organized attack oriented to the right
    - Squares in possession, progression toward the center
    - Circles in defensive organization
    """
    teammates_position = [
        # Wide defense to stretch the opponent's block
        (0.25, 0.75), (0.25, 0.25),
        # Box-to-box midfielders around the center circle
        (0.45, 0.60), (0.45, 0.40),
        # Central midfielder in the middle
        (0.50, 0.50),
        # Forwards near the opponent's zone
        (0.52, 0.60), (0.52, 0.40),
        # Deep support on a wide perimeter
        (0.35, 0.15)
    ]
    
    defenders_position = [
        # Opposing goalkeeper
        (0.95, 0.50),
        # Defensive line recovering position
        (0.75, 0.80), (0.75, 0.60), (0.75, 0.40), (0.75, 0.20),
        # Ball-winning midfielders in front of the defense
        (0.60, 0.70), (0.60, 0.50), (0.60, 0.30)
    ]
    
    # Ball just inside the opponent's half, right of the halfway line
    ball_zone = (0.53, 0.55)
    
    return teammates_position, defenders_position, ball_zone

# Export of the new realistic situations
REALISTIC_SITUATIONS = [
    ("Contre-attaque rapide", situation_contre_attaque_rapide),
    ("Action aile + centre", situation_action_aile_centrage),
    ("Coup franc dangereux", situation_coup_franc_dangereux),
    ("Corner offensif", situation_corner_offensif),
    ("Rupture milieu", situation_rupture_milieu),
    ("Une-deux surface", situation_une_deux_surface),
    ("Pressing haut", situation_pressing_haut),
    ("Sortie balle arrêtée", situation_sortie_balle_arretee),
    ("Attaque organisée droite", situation_attaque_organisee_vers_la_droite)
]
