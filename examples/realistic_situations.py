"""
Situations tactiques réalistes pour tester la fonction goal threat
Basées sur des formations et des scenarios de jeu authentiques
"""

def situation_contre_attaque_rapide():
    """
    Contre-attaque 3v2 classique
    - Équipe attaquante: transition rapide avec 3 joueurs avancés
    - Équipe défensive: seulement 2 défenseurs centraux repositionnés
    - Ball carrier au milieu de terrain avec 2 ailiers en appui
    """
    # Formation attaquante: 4-3-3 en transition offensive
    teammates_position = [
        # Gardien
        (0.05, 0.5),
        # Défenseurs restés derrière  
        (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8),
        # Milieux remontant
        (0.45, 0.35), (0.45, 0.65),
        # Attaquants en contre
        (0.75, 0.3),   # Ailier gauche
        (0.8, 0.5),    # Avant-centre (ball carrier)
        (0.75, 0.7),   # Ailier droit
        (0.7, 0.5)     # Milieu offensif en soutien
    ]
    
    # Défense déséquilibrée - seulement quelques joueurs revenus
    defenders_position = [
        # Gardien adverse
        (0.95, 0.5),
        # Défenseurs centraux
        (0.85, 0.45), (0.85, 0.55),
        # Latéraux pas encore revenus
        (0.6, 0.15), (0.6, 0.85),
        # Milieux défensifs
        (0.7, 0.4), (0.7, 0.6)
    ]
    
    ball_zone = (0.8, 0.5)  # Ballon avec l'avant-centre
    return teammates_position, defenders_position, ball_zone

def situation_action_aile_centrage():
    """
    Action sur l'aile droite avec centre imminent
    - Ailier droit en position de centre
    - Attaquants en position de finition dans la surface
    - Défense organisée mais sous pression
    """
    teammates_position = [
        # Gardien et défense
        (0.05, 0.5),
        (0.2, 0.2), (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
        # Milieux
        (0.4, 0.4), (0.5, 0.5), (0.4, 0.6),
        # Attaquants dans la surface
        (0.88, 0.45),  # Avant-centre
        (0.85, 0.6),   # Second attaquant
        (0.92, 0.15)   # Ailier droit (ball carrier)
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Défense centrale
        (0.9, 0.4), (0.9, 0.6),
        # Latéraux
        (0.85, 0.2), (0.88, 0.8),
        # Milieux défensifs
        (0.75, 0.35), (0.75, 0.55), (0.8, 0.45)
    ]
    
    ball_zone = (0.92, 0.15)  # Ailier en position de centre
    return teammates_position, defenders_position, ball_zone

def situation_coup_franc_dangereux():
    """
    Coup franc à 25m du but
    - Tireur direct face au but
    - Mur défensif organisé
    - Attaquants positionnés pour rebond ou déviation
    """
    teammates_position = [
        # Gardien et défenseurs
        (0.05, 0.5),
        (0.3, 0.3), (0.3, 0.5), (0.3, 0.7),
        # Milieux de terrain
        (0.5, 0.4), (0.5, 0.6),
        # Joueurs dans la surface pour rebond
        (0.88, 0.4), (0.88, 0.6), (0.9, 0.5),
        # Tireur
        (0.75, 0.5),
        # Joueur pour passe courte
        (0.72, 0.4)
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Mur défensif
        (0.82, 0.45), (0.82, 0.48), (0.82, 0.52), (0.82, 0.55),
        # Défenseurs marquant dans la surface
        (0.9, 0.4), (0.9, 0.6), (0.93, 0.5),
        # Milieux défensifs
        (0.75, 0.3), (0.75, 0.7)
    ]
    
    ball_zone = (0.75, 0.5)  # Position du coup franc
    return teammates_position, defenders_position, ball_zone

def situation_corner_offensif():
    """
    Corner offensif avec organisation tactique réaliste
    - Tireur de corner
    - Attaquants positionnés sur les zones de finition
    - Défense en marquage mixte
    """
    teammates_position = [
        # Gardien et défenseur resté derrière
        (0.05, 0.5), (0.5, 0.5),
        # Milieux remontés
        (0.75, 0.3), (0.8, 0.4),
        # Attaquants dans la surface
        (0.92, 0.45),  # Premier poteau
        (0.88, 0.5),   # Centre de la surface
        (0.85, 0.55),  # Second poteau
        (0.82, 0.4),   # Point de penalty
        # Joueurs pour contre-jeu
        (0.7, 0.6), (0.65, 0.35),
        # Tireur de corner
        (1.0, 0.05)
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Défenseurs sur la ligne
        (0.98, 0.45), (0.98, 0.55),
        # Marquage dans la surface
        (0.9, 0.45), (0.86, 0.5), (0.83, 0.55),
        # Défenseurs sur le marquage
        (0.82, 0.4), (0.88, 0.4),
        # Joueur sur poteau arrière
        (0.99, 0.3),
        # Milieu défensif pour dégagement
        (0.75, 0.5)
    ]
    
    ball_zone = (1.0, 0.05)  # Corner droit
    return teammates_position, defenders_position, ball_zone

def situation_rupture_milieu():
    """
    Rupture par le milieu de terrain
    - Milieu offensif porteur du ballon
    - Percussion centrale entre les lignes
    - Attaquants en mouvement pour combinaison
    """
    teammates_position = [
        # Gardien et défense
        (0.05, 0.5),
        (0.2, 0.25), (0.2, 0.4), (0.2, 0.6), (0.2, 0.75),
        # Milieux
        (0.4, 0.3), (0.4, 0.7),  # Milieux excentrés
        # Rupture et attaque
        (0.65, 0.5),   # Milieu offensif (ball carrier)
        (0.8, 0.4),    # Attaquant en appel
        (0.8, 0.6),    # Second attaquant
        (0.75, 0.5)    # Meneur de jeu en soutien
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Défense organisée
        (0.85, 0.3), (0.85, 0.45), (0.85, 0.55), (0.85, 0.7),
        # Milieux défensifs
        (0.7, 0.4), (0.7, 0.6),
        # Milieu récupérateur
        (0.6, 0.5)
    ]
    
    ball_zone = (0.65, 0.5)  # Milieu offensif en rupture
    return teammates_position, defenders_position, ball_zone

def situation_une_deux_surface():
    """
    Une-deux à l'entrée de la surface de réparation
    - Combinaison rapide entre deux attaquants
    - Défense compacte mais prise de vitesse
    - Finition probable
    """
    teammates_position = [
        # Gardien et défense
        (0.05, 0.5),
        (0.25, 0.2), (0.25, 0.4), (0.25, 0.6), (0.25, 0.8),
        # Milieux en soutien
        (0.45, 0.4), (0.45, 0.6), (0.5, 0.5),
        # Combinaison à l'entrée de surface
        (0.83, 0.5),   # Premier attaquant (ball carrier)
        (0.87, 0.45),  # Second attaquant pour remise
        (0.8, 0.6)     # Troisième homme
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Défense resserrée
        (0.88, 0.4), (0.88, 0.6),
        (0.85, 0.45), (0.85, 0.55),
        # Milieux défensifs
        (0.75, 0.35), (0.75, 0.65),
        (0.8, 0.5)
    ]
    
    ball_zone = (0.83, 0.5)  # À l'entrée de la surface
    return teammates_position, defenders_position, ball_zone

def situation_pressing_haut():
    """
    Pressing haut avec récupération du ballon
    - Équipe qui vient de récupérer haut
    - Défense adverse déséquilibrée
    - Transition offensive immédiate
    """
    teammates_position = [
        # Gardien et défense haute
        (0.05, 0.5),
        (0.45, 0.2), (0.45, 0.4), (0.45, 0.6), (0.45, 0.8),
        # Milieux hauts
        (0.6, 0.35), (0.6, 0.65),
        # Pressing et transition
        (0.75, 0.5),   # Récupérateur (ball carrier)
        (0.8, 0.4),    # Attaquant proche
        (0.8, 0.6),    # Second attaquant
        (0.72, 0.5)    # Meneur en soutien
    ]
    
    defenders_position = [
        # Gardien
        (0.95, 0.5),
        # Défense surprise et haute
        (0.7, 0.3), (0.7, 0.7),
        # Défenseurs centraux pas encore repositionnés
        (0.82, 0.45), (0.82, 0.55),
        # Milieux en récupération
        (0.68, 0.4), (0.68, 0.6)
    ]
    
    ball_zone = (0.75, 0.5)  # Ballon récupéré haut
    return teammates_position, defenders_position, ball_zone

def situation_sortie_balle_arretee():
    """
    Sortie de balle arrêtée avec organisation offensive
    - Relance depuis un coup franc défensif
    - Transition organisée vers l'attaque
    - Défense en replacement
    """
    teammates_position = [
        # Gardien
        (0.05, 0.5),
        # Défense étalée pour la relance
        (0.2, 0.2), (0.2, 0.8), (0.15, 0.4), (0.15, 0.6),
        # Milieux en soutien
        (0.35, 0.3), (0.35, 0.7), (0.4, 0.5),
        # Attaquants en mouvement
        (0.65, 0.4), (0.65, 0.6),
        # Relanceur
        (0.25, 0.5)
    ]
    
    defenders_position = [
        # Gardien adverse
        (0.95, 0.5),
        # Défense en replacement
        (0.8, 0.25), (0.8, 0.45), (0.8, 0.55), (0.8, 0.75),
        # Milieux en récupération
        (0.6, 0.4), (0.6, 0.6), (0.7, 0.5)
    ]
    
    ball_zone = (0.25, 0.5)  # Ballon avec le relanceur
    return teammates_position, defenders_position, ball_zone

def situation_attaque_organisee_vers_la_droite():
    """
    Attaque organisée orientée vers la droite
    - Carrés en possession, progression vers l’axe
    - Ronds en organisation défensive
    """
    teammates_position = [
        # Défense large pour étirer le bloc adverse
        (0.25, 0.75), (0.25, 0.25),
        # Milieux relayeurs autour du cercle central
        (0.45, 0.60), (0.45, 0.40),
        # Milieu axial dans l’axe
        (0.50, 0.50),
        # Attaquants proches de la zone adverse
        (0.52, 0.60), (0.52, 0.40),
        # Soutien bas en grand périmètre
        (0.35, 0.15)
    ]
    
    defenders_position = [
        # Gardien adverse
        (0.95, 0.50),
        # Ligne défensive en replacement
        (0.75, 0.80), (0.75, 0.60), (0.75, 0.40), (0.75, 0.20),
        # Milieux récupérateurs devant la défense
        (0.60, 0.70), (0.60, 0.50), (0.60, 0.30)
    ]
    
    # Ballon juste dans la moitié adverse, à droite de la ligne médiane
    ball_zone = (0.53, 0.55)
    
    return teammates_position, defenders_position, ball_zone

# Export des nouvelles situations réalistes
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
