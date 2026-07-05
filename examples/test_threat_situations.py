import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import GOAL_THREAT_MAPS_DIR
from Goal_threat_classes import goal_threat

# Import realistic tactical situations
from realistic_situations import (
    situation_contre_attaque_rapide,
    situation_action_aile_centrage,
    situation_coup_franc_dangereux,
    situation_corner_offensif,
    situation_rupture_milieu,
    situation_une_deux_surface,
    situation_pressing_haut,
    situation_sortie_balle_arretee,
    situation_attaque_organisee_vers_la_droite,
    REALISTIC_SITUATIONS
)

def test_threat_on_typical_situations():
    """Test all realistic tactical situations with professional pitch visualization"""
    
    print("=== THREAT FUNCTION TEST ON REALISTIC TACTICAL SITUATIONS ===\n")
    print("🏆 Situations based on authentic game scenarios")
    print("⚽ Formations and positions from professional football\n")
    
    # Use realistic situations
    situations = REALISTIC_SITUATIONS
    
    # Create output directory for saving images
    output_dir = str(GOAL_THREAT_MAPS_DIR / "threat_analysis_realistic")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Images saved in: {output_dir}/")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 3, figsize=(26, 20))
    fig.patch.set_facecolor('darkgreen')
    fig.suptitle('REALISTIC TACTICAL SITUATIONS ANALYSIS - GOAL THREAT', 
                fontsize=26, fontweight='bold', color='white', y=0.96)
    
    # Flatten axes for easier access
    axes_flat = axes.flatten()
    
    results = []
    
    for i, (name, situation_func) in enumerate(situations):
        teammates_pos, defenders_pos, ball_pos = situation_func()
        
        print(f"🎯 {name.upper()}:")
        print(f"   📍 Ball position: {ball_pos}")
        print(f"   👥 Teammates: {len(teammates_pos)} | 🛡️  Defenders: {len(defenders_pos)}")
        
        # Calculate threat with fewer passes for speed
        print(f"   🔄 Calculating...")
        threat = goal_threat(
            ball_zone=ball_pos,
            teammates_position=teammates_pos,
            defenders_position=defenders_pos,
            passes_restantes=2  # Reduced to 2 for speed!
        )
        
        results.append((name, threat))
        
        # Level classification
        if threat > 0.6:
            level_emoji = "🔴"
            level_text = "CRITICAL"
        elif threat > 0.4:
            level_emoji = "🟠"
            level_text = "HIGH"
        elif threat > 0.25:
            level_emoji = "🟡"
            level_text = "MEDIUM"
        else:
            level_emoji = "🟢"
            level_text = "LOW"
        
        print(f"   ⚡ Threat level: {threat:.4f} {level_emoji} ({level_text})\n")
        
        # Draw pitch for this situation
        if i < len(axes_flat):
            # Individual save path for high-resolution image
            individual_save_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_').replace('+', '_')}.png")
            
            # Create individual figure for high-resolution saving
            fig_individual, ax_individual = plt.subplots(1, 1, figsize=(18, 14))
            fig_individual.patch.set_facecolor('darkgreen')
            
            draw_football_pitch(ax_individual, ball_pos, teammates_pos, defenders_pos, 
                              threat, f"{name.upper()} - REALISTIC SITUATION", 
                              save_path=individual_save_path)
            
            plt.close(fig_individual)  # Close individual figure
            
            # Also draw on main grid (simplified version)
            draw_football_pitch(axes_flat[i], ball_pos, teammates_pos, defenders_pos, 
                              threat, name)
    
    # Hide unused axes
    for j in range(len(situations), len(axes_flat)):
        axes_flat[j].axis('off')
        axes_flat[j].set_facecolor('darkgreen')
    
    # Create summary chart in last subplot
    if len(situations) < len(axes_flat):
        # Bar chart of threat levels
        ax_summary = axes_flat[-1]
        ax_summary.set_facecolor('black')
        
        names = [r[0] for r in results]
        threats = [r[1] for r in results]
        
        # Colors according to threat level (adapted thresholds)
        colors = ['darkred' if t > 0.6 else 'red' if t > 0.4 else 'orange' if t > 0.25 else 'green' 
                 for t in threats]
        
        bars = ax_summary.bar(range(len(names)), threats, color=colors, 
                             edgecolor='white', linewidth=2, alpha=0.9)
        
        ax_summary.set_title('THREAT LEVELS COMPARISON', 
                           fontsize=16, fontweight='bold', color='white')
        ax_summary.set_ylabel('Threat Level', fontsize=14, color='white')
        ax_summary.set_xticks(range(len(names)))
        ax_summary.set_xticklabels(names, rotation=45, ha='right', color='white', fontsize=10)
        ax_summary.tick_params(colors='white')
        ax_summary.grid(True, alpha=0.3, color='white')
        
        # Add values on bars
        for bar, threat in zip(bars, threats):
            height = bar.get_height()
            ax_summary.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{threat:.3f}', ha='center', va='bottom', 
                           fontweight='bold', color='white', fontsize=9)
    
    # Save complete grid
    grid_save_path = os.path.join(output_dir, "threat_analysis_realistic_complete.png")
    plt.tight_layout()
    plt.savefig(grid_save_path, dpi=300, bbox_inches='tight', facecolor='darkgreen')
    print(f"💾 Complete grid saved: {grid_save_path}")
    
    plt.show()
    
    # Textual summary with tactical analysis
    print("\n" + "="*70)
    print("🎯 TACTICAL ANALYSIS OF REALISTIC SITUATIONS")
    print("="*70)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 RANKING BY DANGER LEVEL:")
    for i, (name, threat) in enumerate(sorted_results, 1):
        if threat > 0.6:
            level_emoji = "🔴"
            threat_level = "CRITICAL"
        elif threat > 0.4:
            level_emoji = "🟠"  
            threat_level = "HIGH"
        elif threat > 0.25:
            level_emoji = "🟡"
            threat_level = "MEDIUM"
        else:
            level_emoji = "🟢"
            threat_level = "LOW"
        
        print(f"{i:2d}. {level_emoji} {name:25s}: {threat:.4f} ({threat_level})")
    
    # Analysis by situation type
    print(f"\n📊 ANALYSIS BY SITUATION TYPE:")
    transition_situations = [r for r in results if any(word in r[0].lower() for word in ['counter', 'pressing', 'breakaway'])]
    set_piece_situations = [r for r in results if any(word in r[0].lower() for word in ['free kick', 'corner', 'set piece'])]
    open_play_situations = [r for r in results if r not in transition_situations and r not in set_piece_situations]
    
    if transition_situations:
        avg_transition = np.mean([r[1] for r in transition_situations])
        print(f"   ⚡ Transitions (counter-attacks, pressing): {avg_transition:.3f}")
    
    if set_piece_situations:
        avg_set_piece = np.mean([r[1] for r in set_piece_situations])
        print(f"   🎯 Set pieces (FK, corners, etc.): {avg_set_piece:.3f}")
    
    if open_play_situations:
        avg_open_play = np.mean([r[1] for r in open_play_situations])
        print(f"   ⚽ Open play (combinations, crosses): {avg_open_play:.3f}")
    
    print(f"\n📈 GLOBAL STATISTICS:")
    print(f"   • Most dangerous situation : {sorted_results[0][0]} ({sorted_results[0][1]:.4f})")
    print(f"   • Least dangerous situation: {sorted_results[-1][0]} ({sorted_results[-1][1]:.4f})")
    print(f"   • Average threat          : {np.mean([r[1] for r in results]):.4f}")
    print(f"   • Standard deviation      : {np.std([r[1] for r in results]):.4f}")
    
    critical_situations = len([r for r in results if r[1] > 0.6])
    high_situations = len([r for r in results if 0.4 < r[1] <= 0.6])
    
    print(f"\n🚨 LEVEL DISTRIBUTION:")
    print(f"   • Critical situations (>0.6) : {critical_situations}/{len(results)}")
    print(f"   • High situations (0.4-0.6)  : {high_situations}/{len(results)}")
    
    print(f"\n💾 {len(situations)+1} high resolution images saved in '{output_dir}/'")
    print("📋 Each image represents an authentic tactical situation from professional football")
    
    return results

def test_single_situation(situation_name):
    """Test a specific situation with detailed visualization and saving"""
    situations_dict = {
        "contre_attaque": ("Fast counter-attack", situation_contre_attaque_rapide),
        "aile_centre": ("Wing action + cross", situation_action_aile_centrage),
        "coup_franc": ("Dangerous free kick", situation_coup_franc_dangereux),
        "corner": ("Offensive corner", situation_corner_offensif),
        "rupture": ("Midfield breakthrough", situation_rupture_milieu),
        "une_deux": ("One-two in box", situation_une_deux_surface),
        "pressing": ("High pressing", situation_pressing_haut),
        "balle_arretee": ("Set piece recovery", situation_sortie_balle_arretee),
        "attaque_droite": ("Organized right attack", situation_attaque_organisee_vers_la_droite)
    }
    
    if situation_name not in situations_dict:
        print(f"❌ Situation '{situation_name}' not found!")
        print(f"📋 Available realistic situations:")
        for key, (name, _) in situations_dict.items():
            print(f"   • {key:15s}: {name}")
        return
    
    name, situation_func = situations_dict[situation_name]
    teammates_pos, defenders_pos, ball_pos = situation_func()
    
    print(f"🔍 DETAILED ANALYSIS: {name.upper()}")
    print("="*60)
    print("⚽ Situation based on realistic game scenario")
    
    # Calculate threat with reduced depth for speed
    print("🔄 Calculating threat (fast mode)...")
    threat = goal_threat(
        ball_zone=ball_pos,
        teammates_position=teammates_pos,
        defenders_position=defenders_pos,
        passes_restantes=2  # Reduced for speed
    )
    
    # Create save directory
    output_dir = str(GOAL_THREAT_MAPS_DIR / "threat_analysis_detailed_realistic")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed high-resolution visualization
    fig, ax = plt.subplots(1, 1, figsize=(22, 16))
    fig.patch.set_facecolor('darkgreen')
    
    # Save path
    save_path = os.path.join(output_dir, f"{situation_name}_detailed_realistic.png")
    
    # Draw pitch with all details
    draw_football_pitch(ax, ball_pos, teammates_pos, defenders_pos, threat, 
                       f"{name.upper()} - DETAILED TACTICAL ANALYSIS", 
                       save_path=save_path)
    
    plt.tight_layout()
    plt.show()
    
    # Detailed textual analysis with tactical context
    print(f"\n� DONNÉES DE LA SITUATION:")
    print(f"   �📍 Position du ballon: ({ball_pos[0]:.3f}, {ball_pos[1]:.3f})")
    print(f"   👥 Joueurs attaquants: {len(teammates_pos)}")
    print(f"   🛡️  Joueurs défenseurs: {len(defenders_pos)}")
    print(f"   ⚡ Niveau de menace: {threat:.4f}")
    
    # Classify the threat level with thresholds adapted to realism
    if threat > 0.6:
        level_emoji = "🔴"
        threat_level = "CRITIQUE"
        tactical_advice = "Situation extrêmement dangereuse! Intervention défensive urgente requise."
    elif threat > 0.4:
        level_emoji = "🟠"
        threat_level = "ÉLEVÉ"
        tactical_advice = "Situation très menaçante. Resserrement défensif et anticipation nécessaires."
    elif threat > 0.25:
        level_emoji = "🟡"
        threat_level = "MOYEN"
        tactical_advice = "Situation sous contrôle mais vigilance requise sur les passes clés."
    else:
        level_emoji = "🟢"
        threat_level = "FAIBLE"
        tactical_advice = "Situation défensivement maîtrisée. Opportunité de transition."
    
    print(f"   {level_emoji} Évaluation: {threat_level}")
    print(f"   💡 Conseil tactique: {tactical_advice}")
    
    # Geometric and tactical analysis
    print(f"\n🎯 ANALYSE TACTIQUE AVANCÉE:")
    
    # Distance to goal and shot angle
    goal_pos = (1.0, 0.5)
    distance_to_goal = np.sqrt((ball_pos[0] - goal_pos[0])**2 + (ball_pos[1] - goal_pos[1])**2)
    print(f"   � Distance au but: {distance_to_goal:.3f} ({distance_to_goal*105:.1f}m)")
    
    # Shot angle analysis
    angle_shot = abs(ball_pos[1] - 0.5)  # Offset from the center
    if angle_shot < 0.1:
        angle_quality = "Excellent (face au but)"
    elif angle_shot < 0.2:
        angle_quality = "Bon (angle favorable)"
    else:
        angle_quality = "Difficile (angle fermé)"
    print(f"   📐 Qualité de l'angle: {angle_quality}")
    
    # Defensive analysis
    if defenders_pos:
        defenders_array = np.array(defenders_pos)
        defenders_center = np.mean(defenders_array, axis=0)
        print(f"   🛡️  Centre défensif: ({defenders_center[0]:.3f}, {defenders_center[1]:.3f})")
        
        # Defensive compactness
        defenders_distances = [np.linalg.norm(pos - defenders_center) for pos in defenders_pos]
        avg_defensive_spread = np.mean(defenders_distances)
        
        if avg_defensive_spread < 0.1:
            defensive_quality = "Très compacte"
        elif avg_defensive_spread < 0.15:
            defensive_quality = "Compacte"
        else:
            defensive_quality = "Étalée"
        
        print(f"   📏 Organisation défensive: {defensive_quality} (étalement: {avg_defensive_spread:.3f})")
        
        # Defenders in the critical zone
        critical_zone_defenders = len([pos for pos in defenders_pos if pos[0] > 0.8])
        print(f"   🚨 Défenseurs en zone critique: {critical_zone_defenders}")
    
    # Offensive analysis
    if teammates_pos:
        teammates_array = np.array(teammates_pos)
        teammates_center = np.mean(teammates_array, axis=0)
        print(f"   ⚔️  Centre offensif: ({teammates_center[0]:.3f}, {teammates_center[1]:.3f})")
        
        # Players in the opponent's box
        attackers_in_box = len([pos for pos in teammates_pos if pos[0] > 0.83])
        print(f"   🏃 Attaquants dans la surface: {attackers_in_box}")
        
        # Offensive support
        support_players = len([pos for pos in teammates_pos if 0.6 < pos[0] < 0.83])
        print(f"   🤝 Joueurs de soutien: {support_players}")
    
    # Contextual analysis by situation type
    print(f"\n� CONTEXTE TACTIQUE:")
    situation_context = {
        "contre_attaque": "Transition rapide exploitant le déséquilibre défensif adverse",
        "aile_centre": "Action latérale avec centre imminent vers la surface",
        "coup_franc": "Balle arrêtée offrant une frappe directe ou combinaison",
        "corner": "Situation de balle arrêtée avec avantage numérique offensif",
        "rupture": "Percussion centrale entre les lignes défensives",
        "une_deux": "Combinaison rapide à l'entrée de la surface de réparation",
        "pressing": "Récupération haute du ballon avec défense adverse déséquilibrée",
        "balle_arretee": "Relance organisée depuis une position défensive",
        "attaque_droite": "Attaque organisée avec progression vers le côté droit du terrain"
    }
    
    context = situation_context.get(situation_name, "Situation de jeu standard")
    print(f"   📝 Description: {context}")
    
    # Specific tactical recommendations
    recommendations = {
        "contre_attaque": "Privilégier la verticalité et la vitesse d'exécution",
        "aile_centre": "Timing du centre crucial, viser les zones de finition",
        "coup_franc": "Choix entre frappe directe et combinaison selon le mur",
        "corner": "Exploiter les zones de finition, attention au marquage",
        "rupture": "Enchaîner rapidement avec les appels des attaquants",
        "une_deux": "Synchronisation parfaite requise pour la remise",
        "pressing": "Transition immédiate avant le replacement défensif",
        "balle_arretee": "Progression méthodique en évitant la perte de balle",
        "attaque_droite": "Maintenir la largeur et créer des décalages sur le côté fort"
    }
    
    recommendation = recommendations.get(situation_name, "Jouer selon les principes tactiques de base")
    print(f"   💭 Recommandation: {recommendation}")
    
    print(f"\n�💾 Image haute résolution sauvegardée: {save_path}")
    print("🎯 Cette analyse se base sur une situation tactique authentique du football moderne")
    
    return threat

def draw_professional_pitch(ax, pitch_length=105, pitch_width=68):
    """Draw a professional football pitch (Animation_terrain.py style)"""
    # Green background
    ax.set_facecolor("forestgreen")
    ax.figure.patch.set_facecolor("forestgreen")
    
    # Pitch outline
    ax.add_patch(patches.Rectangle(
        (0, 0), pitch_length, pitch_width,
        fill=False, edgecolor="white", linewidth=3
    ))
    
    # Halfway line
    ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], color="white", linewidth=3)
    
    # Center circle
    center = (pitch_length/2, pitch_width/2)
    ax.add_patch(patches.Circle(center, 9.15, fill=False, edgecolor="white", linewidth=2))
    ax.plot(center[0], center[1], 'wo', markersize=4)
    
    # Penalty area left (our goal)
    penalty_left = patches.Rectangle((0, (pitch_width-40.32)/2), 16.5, 40.32, 
                                   fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(penalty_left)
    
    # Goal area left
    small_left = patches.Rectangle((0, (pitch_width-18.32)/2), 5.5, 18.32,
                                 fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(small_left)
    
    # Penalty area right (opponent's goal)
    penalty_right = patches.Rectangle((pitch_length-16.5, (pitch_width-40.32)/2), 16.5, 40.32,
                                    fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(penalty_right)
    
    # Goal area right
    small_right = patches.Rectangle((pitch_length-5.5, (pitch_width-18.32)/2), 5.5, 18.32,
                                  fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(small_right)
    
    # Goals
    goal_height = 7.32
    goal_y = (pitch_width - goal_height) / 2
    
    # Left goal (our goal)
    goal_left = patches.Rectangle((-2, goal_y), 2, goal_height,
                                fill=True, facecolor="lightblue", edgecolor="white", linewidth=3)
    ax.add_patch(goal_left)
    
    # Right goal (opponent's goal)
    goal_right = patches.Rectangle((pitch_length, goal_y), 2, goal_height,
                                 fill=True, facecolor="lightcoral", edgecolor="white", linewidth=3)
    ax.add_patch(goal_right)
    
    # Penalty spots
    ax.plot(11, pitch_width/2, 'wo', markersize=6)
    ax.plot(pitch_length-11, pitch_width/2, 'wo', markersize=6)
    
    # Penalty arcs
    penalty_arc_left = patches.Arc((11, pitch_width/2), 18.3, 18.3, 
                                 theta1=308, theta2=52, color="white", linewidth=2)
    ax.add_patch(penalty_arc_left)
    
    penalty_arc_right = patches.Arc((pitch_length-11, pitch_width/2), 18.3, 18.3,
                                  theta1=128, theta2=232, color="white", linewidth=2)
    ax.add_patch(penalty_arc_right)
    
    # Axis configuration
    ax.set_xlim(-5, pitch_length+5)
    ax.set_ylim(-5, pitch_width+5)
    ax.set_aspect("equal")
    ax.axis("off")

def draw_threat_gauge(ax, x, y, width, height, value, color, label, max_value=1.0):
    """Draw a professional threat gauge"""
    # Gauge background
    bg_rect = patches.Rectangle((x, y), width, height, 
                              facecolor='black', edgecolor='white', linewidth=2, alpha=0.7)
    ax.add_patch(bg_rect)
    
    # Progress bar
    progress_width = width * (value / max_value)
    if progress_width > 0:
        progress_rect = patches.Rectangle((x, y), progress_width, height,
                                        facecolor=color, alpha=0.8)
        ax.add_patch(progress_rect)
    
    # Gauge text
    ax.text(x + width/2, y + height/2, f'{value*100:.1f}%',
           ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Label
    ax.text(x, y + height + 1, label, fontsize=12, fontweight='bold', color='white')
    
    return [bg_rect, progress_rect] if progress_width > 0 else [bg_rect]

def convert_normalized_to_pitch(pos, pitch_length=105, pitch_width=68):
    """Convert normalized coordinates (0-1) to the pitch dimensions"""
    x, y = pos
    return (x * pitch_length, y * pitch_width)

def draw_football_pitch(ax, ball_pos, teammates_pos, defenders_pos, threat_value, title, save_path=None):
    """Draw a professional football pitch with the players and the ball"""
    
    # Clear the axis
    ax.clear()
    
    # Real pitch dimensions
    pitch_length, pitch_width = 105, 68
    
    # Draw the professional pitch
    draw_professional_pitch(ax, pitch_length, pitch_width)
    
    # Convert the normalized positions to the pitch dimensions
    ball_pitch = convert_normalized_to_pitch(ball_pos, pitch_length, pitch_width)
    teammates_pitch = [convert_normalized_to_pitch(pos, pitch_length, pitch_width) for pos in teammates_pos]
    defenders_pitch = [convert_normalized_to_pitch(pos, pitch_length, pitch_width) for pos in defenders_pos]
    
    # Draw the players with a professional style
    # Teammates (attacking team) in blue
    for i, (x, y) in enumerate(teammates_pitch):
        circle = patches.Circle((x, y), 1.5, facecolor='royalblue', edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.annotate(f'{i+1}', (x, y), ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')
    
    # Defenders (defending team) in red
    for i, (x, y) in enumerate(defenders_pitch):
        square = patches.Rectangle((x-1.5, y-1.5), 3, 3, facecolor='firebrick', 
                                 edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(square)
        ax.annotate(f'{i+1}', (x, y), ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')
    
    # Ball with a glossy effect
    ball_circle = patches.Circle(ball_pitch, 1.2, facecolor='gold', edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(ball_circle)
    # Glossy effect on the ball
    highlight = patches.Circle((ball_pitch[0]-0.3, ball_pitch[1]+0.3), 0.4, 
                             facecolor='white', alpha=0.6, zorder=11)
    ax.add_patch(highlight)
    
    # Professional threat gauge
    threat_color = 'red' if threat_value > 0.5 else 'orange' if threat_value > 0.3 else 'green'
    gauge_elements = draw_threat_gauge(ax, 5, pitch_width-8, 25, 4, threat_value, threat_color, 
                                     'GOAL THREAT')
    
    # Professional title with background
    title_bg = patches.Rectangle((pitch_length/2-25, pitch_width+8), 50, 6,
                               facecolor='black', alpha=0.8, edgecolor='white', linewidth=2)
    ax.add_patch(title_bg)
    ax.text(pitch_length/2, pitch_width+11, title, ha='center', va='center',
           fontsize=16, fontweight='bold', color='white')
    
    # Tactical information
    info_text = f'Ball: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f})\n'
    info_text += f'Teammates: {len(teammates_pos)} | Defenders: {len(defenders_pos)}\n'
    info_text += f'Threat: {threat_value:.4f}'
    
    ax.text(5, 5, info_text, fontsize=10, color='white', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, 
                    edgecolor='white', linewidth=1))
    
    # Professional legend
    legend_x = pitch_length - 30
    legend_y = pitch_width - 15
    
    # Legend background
    legend_bg = patches.Rectangle((legend_x-2, legend_y-2), 28, 12,
                                facecolor='black', alpha=0.8, edgecolor='white', linewidth=1)
    ax.add_patch(legend_bg)
    
    # Legend elements
    # Teammates
    teammate_circle = patches.Circle((legend_x, legend_y+8), 1, facecolor='royalblue', 
                                   edgecolor='white', linewidth=1)
    ax.add_patch(teammate_circle)
    ax.text(legend_x+3, legend_y+8, 'Teammates', va='center', fontsize=10, 
           color='white', fontweight='bold')
    
    # Defenders
    defender_square = patches.Rectangle((legend_x-1, legend_y+4), 2, 2, facecolor='firebrick',
                                      edgecolor='white', linewidth=1)
    ax.add_patch(defender_square)
    ax.text(legend_x+3, legend_y+5, 'Defenders', va='center', fontsize=10,
           color='white', fontweight='bold')
    
    # Ball
    ball_legend = patches.Circle((legend_x, legend_y+1), 0.8, facecolor='gold',
                               edgecolor='black', linewidth=1)
    ax.add_patch(ball_legend)
    ax.text(legend_x+3, legend_y+1, 'Ball', va='center', fontsize=10,
           color='white', fontweight='bold')
    
    # Save image if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='forestgreen',
                   edgecolor='none', pad_inches=0.1)
        print(f"💾 Image saved: {save_path}")
    
    return gauge_elements

def test_threat_quick():
    """Ultra-fast version for development tests"""
    print("=== QUICK THREAT FUNCTION TEST ===\n")
    print("⚡ Development mode - Simplified calculations for speed\n")
    
    # Only 3 representative situations
    quick_situations = [
        ("Fast counter-attack", situation_contre_attaque_rapide),
        ("One-two in box", situation_une_deux_surface),
        ("Offensive corner", situation_corner_offensif)
    ]
    
    results = []
    
    for i, (name, situation_func) in enumerate(quick_situations):
        teammates_pos, defenders_pos, ball_pos = situation_func()
        
        print(f"🎯 {name}:")
        print(f"   📍 Ball: {ball_pos}")
        print(f"   👥 {len(teammates_pos)} vs 🛡️ {len(defenders_pos)}")
        print(f"   🔄 Simplified calculation...")
        
        # Ultra-fast calculation with 0 passes (only direct shot)
        threat = goal_threat(
            ball_zone=ball_pos,
            teammates_position=teammates_pos,
            defenders_position=defenders_pos,
            passes_restantes=0  # Instant calculation!
        )
        
        results.append((name, threat))
        
        # Quick classification
        if threat > 0.5:
            level = "🔴 HIGH"
        elif threat > 0.3:
            level = "🟡 MEDIUM"
        else:
            level = "🟢 LOW"
        
        print(f"   ⚡ Threat: {threat:.3f} {level}\n")
    
    # Quick summary
    print("📊 QUICK SUMMARY:")
    for i, (name, threat) in enumerate(sorted(results, key=lambda x: x[1], reverse=True), 1):
        print(f"   {i}. {name}: {threat:.3f}")
    
    return results

def test_threat_with_options():
    """Test with speed options"""
    print("=== AVAILABLE TEST OPTIONS ===\n")
    print("1. 🏃 Ultra-fast test (3 situations, 0 passes)")
    print("2. ⚡ Fast test (all situations, 1 pass)")
    print("3. 🐌 Complete test (all situations, 2 passes)")
    print("4. 🔥 Detailed test (all situations, 3 passes + images)\n")
    
    try:
        choice = input("Your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🏃 LAUNCHING ULTRA-FAST TEST...")
            return test_threat_quick()
            
        elif choice == "2":
            print("\n⚡ LAUNCHING FAST TEST...")
            return test_threat_fast()
            
        elif choice == "3":
            print("\n🐌 LAUNCHING COMPLETE TEST...")
            return test_threat_complete()
            
        elif choice == "4":
            print("\n🔥 LAUNCHING DETAILED TEST...")
            return test_threat_on_typical_situations()
            
        else:
            print("❌ Invalid choice, launching fast test by default...")
            return test_threat_fast()
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return []

def test_threat_fast():
    """Fast version without images"""
    print("=== FAST TEST OF REALISTIC SITUATIONS ===\n")
    
    results = []
    total_situations = len(REALISTIC_SITUATIONS)
    
    for i, (name, situation_func) in enumerate(REALISTIC_SITUATIONS, 1):
        teammates_pos, defenders_pos, ball_pos = situation_func()
        
        print(f"[{i}/{total_situations}] {name}... ", end="", flush=True)
        
        # Fast calculation with 1 pass only
        threat = goal_threat(
            ball_zone=ball_pos,
            teammates_position=teammates_pos,
            defenders_position=defenders_pos,
            passes_restantes=1
        )
        
        results.append((name, threat))
        print(f"✅ {threat:.3f}")
    
    # Quick summary
    print(f"\n📊 RESULTS (ranked by danger level):")
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    for i, (name, threat) in enumerate(sorted_results, 1):
        if threat > 0.4:
            emoji = "🔴"
        elif threat > 0.25:
            emoji = "🟡"
        else:
            emoji = "🟢"
        print(f"{i:2d}. {emoji} {name:25s}: {threat:.4f}")
    
    avg_threat = np.mean([r[1] for r in results])
    print(f"\n⚡ Average threat: {avg_threat:.3f}")
    print(f"🏆 Most dangerous: {sorted_results[0][0]} ({sorted_results[0][1]:.3f})")
    
    return results

def test_threat_complete():
    """Complete version without images"""
    print("=== COMPLETE TEST OF REALISTIC SITUATIONS ===\n")
    
    results = []
    total_situations = len(REALISTIC_SITUATIONS)
    
    print(f"🔄 Analyzing {total_situations} situations with 2 passes...")
    
    for i, (name, situation_func) in enumerate(REALISTIC_SITUATIONS, 1):
        teammates_pos, defenders_pos, ball_pos = situation_func()
        
        print(f"[{i}/{total_situations}] 🎯 {name}")
        print(f"   📍 Ball: {ball_pos}")
        print(f"   👥 {len(teammates_pos)} vs 🛡️ {len(defenders_pos)}")
        print(f"   🔄 Calculating... ", end="", flush=True)
        
        # Calculation with 2 passes (speed/precision compromise)
        threat = goal_threat(
            ball_zone=ball_pos,
            teammates_position=teammates_pos,
            defenders_position=defenders_pos,
            passes_restantes=2
        )
        
        results.append((name, threat))
        
        # Classification
        if threat > 0.5:
            level = "🔴 CRITICAL"
        elif threat > 0.3:
            level = "🟠 HIGH"
        elif threat > 0.2:
            level = "🟡 MEDIUM"
        else:
            level = "🟢 LOW"
        
        print(f"✅ {threat:.4f} ({level})\n")
    
    # Complete analysis
    print("="*60)
    print("🎯 COMPLETE RESULTS ANALYSIS")
    print("="*60)
    
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 FINAL RANKING:")
    for i, (name, threat) in enumerate(sorted_results, 1):
        if threat > 0.5:
            emoji = "🔴"
            level = "CRITICAL"
        elif threat > 0.3:
            emoji = "🟠"
            level = "HIGH"
        elif threat > 0.2:
            emoji = "🟡"
            level = "MEDIUM"
        else:
            emoji = "🟢"
            level = "LOW"
        
        print(f"{i:2d}. {emoji} {name:25s}: {threat:.4f} ({level})")
    
    # Statistics
    threats = [r[1] for r in results]
    avg_threat = np.mean(threats)
    std_threat = np.std(threats)
    
    print(f"\n📈 STATISTICS:")
    print(f"   • Average threat    : {avg_threat:.4f}")
    print(f"   • Standard deviation: {std_threat:.4f}")
    print(f"   • Maximum threat    : {max(threats):.4f}")
    print(f"   • Minimum threat    : {min(threats):.4f}")
    
    # Distribution by level
    critical = len([t for t in threats if t > 0.5])
    high = len([t for t in threats if 0.3 < t <= 0.5])
    medium = len([t for t in threats if 0.2 < t <= 0.3])
    low = len([t for t in threats if t <= 0.2])
    
    print(f"\n📊 DISTRIBUTION:")
    print(f"   • Critical (>0.5)   : {critical}/{total_situations}")
    print(f"   • High (0.3-0.5)    : {high}/{total_situations}")
    print(f"   • Medium (0.2-0.3)  : {medium}/{total_situations}")
    print(f"   • Low (≤0.2)        : {low}/{total_situations}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tactical situations with visualization')
    parser.add_argument('--situation', '-s', type=str, help='Test a specific situation')
    parser.add_argument('--quick', '-q', action='store_true', help='Ultra-fast test (3 situations)')
    parser.add_argument('--fast', '-f', action='store_true', help='Fast test (no images)')
    parser.add_argument('--complete', '-c', action='store_true', help='Complete test (no images)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode with choice')
    
    args = parser.parse_args()
    
    if args.situation:
        test_single_situation(args.situation)
    elif args.quick:
        print("🏃 ULTRA-FAST MODE ACTIVATED")
        test_threat_quick()
    elif args.fast:
        print("⚡ FAST MODE ACTIVATED")
        test_threat_fast()
    elif args.complete:
        print("🐌 COMPLETE MODE ACTIVATED")
        test_threat_complete()
    elif args.interactive:
        test_threat_with_options()
    else:
        # By default, propose options
        print("🎯 TACTICAL SITUATIONS ANALYZER")
        print("="*50)
        print("Choose your analysis mode:")
        print()
        print("Available arguments:")
        print("  -q, --quick      : Ultra-fast test (3 situations)")
        print("  -f, --fast       : Fast test (all situations, no images)")
        print("  -c, --complete   : Complete test (no images)")
        print("  -i, --interactive: Interactive mode")
        print("  -s SITUATION     : Specific situation")
        print()
        print("Without argument: Test with images (SLOW)")
        print()
        
        choice = input("Launch interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            test_threat_with_options()
        else:
            print("ℹ️  Use -q for quick test or -h for help")
            test_threat_fast()  # Fast mode by default
