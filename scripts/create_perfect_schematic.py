#!/usr/bin/env python3
"""
Create the perfect ZeroProofML schematic with proper layout and precise arrows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
import numpy as np
from pathlib import Path

# Configure matplotlib
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = False

def create_perfect_schematic():
    """Create the perfect schematic with proper spacing and precise arrows"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Professional color palette
    colors = {
        'input': '#E3F2FD',      # Light blue
        'tr_layer': '#E8F5E9',   # Light green  
        'guard': '#FFF8E1',      # Light yellow
        'real': '#FFEBEE',       # Light red
        'output': '#F3E5F5',     # Light purple
        'decision': '#FFE0B2',   # Light orange
    }
    
    # Define precise positions
    input_x, input_y = 2, 4
    tr_x, tr_y = 6, 4
    decision_x, decision_y = 9.5, 4
    guard_x, guard_y = 12.5, 5.5
    real_x, real_y = 12.5, 2.5
    output_x, output_y = 16, 4
    
    # 1. Input Box
    input_box = FancyBboxPatch((input_x-1, input_y-0.5), 2, 1, 
                              boxstyle="round,pad=0.08",
                              facecolor=colors['input'], 
                              edgecolor='#1976D2', 
                              linewidth=2)
    ax.add_patch(input_box)
    ax.text(input_x, input_y, 'Input', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(input_x, input_y-0.3, 'x ∈ ℝⁿ', ha='center', va='center', fontsize=10)
    
    # 2. TR-Rational Layer
    tr_box = FancyBboxPatch((tr_x-1.5, tr_y-0.75), 3, 1.5, 
                           boxstyle="round,pad=0.08",
                           facecolor=colors['tr_layer'], 
                           edgecolor='#388E3C', 
                           linewidth=2)
    ax.add_patch(tr_box)
    ax.text(tr_x, tr_y+0.3, 'TR-Rational Layer', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(tr_x, tr_y, 'P(x) / Q(x)', ha='center', va='center', fontsize=11)
    ax.text(tr_x, tr_y-0.3, 'with transreal tags', ha='center', va='center',
            fontsize=9, style='italic', color='#666')
    
    # 3. Decision Diamond (as rotated square)
    diamond_size = 0.5
    diamond = Polygon([
        [decision_x, decision_y + diamond_size],  # top
        [decision_x + diamond_size, decision_y],  # right
        [decision_x, decision_y - diamond_size],  # bottom
        [decision_x - diamond_size, decision_y]   # left
    ], facecolor=colors['decision'], edgecolor='#FF6F00', linewidth=2)
    ax.add_patch(diamond)
    # Single-line condition using mathtext to avoid line breaks
    ax.text(
        decision_x,
        decision_y,
        r'$|Q|>\tau$',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
    )
    
    # 4. Guard Mode (top path)
    guard_box = FancyBboxPatch((guard_x-1.25, guard_y-0.4), 2.5, 0.8,
                              boxstyle="round,pad=0.08",
                              facecolor=colors['guard'],
                              edgecolor='#FF8F00',
                              linewidth=2)
    ax.add_patch(guard_box)
    ax.text(guard_x, guard_y, 'Guard Mode', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(guard_x, guard_y-0.25, 'Standard P/Q', ha='center', va='center', fontsize=9)
    ax.text(guard_x+1.5, guard_y, '→ REAL tag', ha='center', va='center',
            fontsize=8, color='#1565C0', style='italic')
    
    # 5. Real Mode (bottom path)
    real_box = FancyBboxPatch((real_x-1.25, real_y-0.4), 2.5, 0.8,
                             boxstyle="round,pad=0.08",
                             facecolor=colors['real'],
                             edgecolor='#D32F2F',
                             linewidth=2)
    ax.add_patch(real_box)
    ax.text(real_x, real_y, 'Real Mode', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(real_x, real_y-0.25, '±∞ or Φ', ha='center', va='center', fontsize=9)
    ax.text(real_x+1.5, real_y, '→ INF/NULL tags', ha='center', va='center',
            fontsize=8, color='#D32F2F', style='italic')
    
    # 6. Output Box
    output_box = FancyBboxPatch((output_x-1, output_y-0.5), 2, 1,
                               boxstyle="round,pad=0.08",
                               facecolor=colors['output'],
                               edgecolor='#7B1FA2',
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(output_x, output_y, 'Output', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(output_x, output_y-0.3, '(y, tag) ∈ 𝕋', ha='center', va='center', fontsize=10)
    
    # Precise arrows with better styling
    arrow_style = dict(arrowstyle='->', lw=2, color='black')
    
    # Input → TR-Rational
    ax.annotate('', xy=(tr_x-1.5, tr_y), xytext=(input_x+1, input_y),
                arrowprops=arrow_style)
    
    # TR-Rational → Decision
    ax.annotate('', xy=(decision_x-0.5, decision_y), xytext=(tr_x+1.5, tr_y),
                arrowprops=arrow_style)
    
    # Decision → Guard (YES)
    ax.annotate('', xy=(guard_x-1.25, guard_y-0.2), xytext=(decision_x+0.3, decision_y+0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#388E3C',
                               connectionstyle="arc3,rad=0.2"))
    ax.text(decision_x+0.8, decision_y+0.8, 'YES', fontsize=9, color='#388E3C',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white', 
                     edgecolor='#388E3C', linewidth=1))
    
    # Decision → Real (NO)
    ax.annotate('', xy=(real_x-1.25, real_y+0.2), xytext=(decision_x+0.3, decision_y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#D32F2F',
                               connectionstyle="arc3,rad=-0.2"))
    ax.text(decision_x+0.8, decision_y-0.8, 'NO', fontsize=9, color='#D32F2F',
            fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white',
                     edgecolor='#D32F2F', linewidth=1))
    
    # Guard → Output
    ax.annotate('', xy=(output_x-1, output_y+0.2), xytext=(guard_x+1.25, guard_y-0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#FF8F00',
                               connectionstyle="arc3,rad=-0.2"))
    
    # Real → Output
    ax.annotate('', xy=(output_x-1, output_y-0.2), xytext=(real_x+1.25, real_y+0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#D32F2F',
                               connectionstyle="arc3,rad=0.2"))
    
    # Add threshold notation below decision
    ax.text(decision_x, decision_y-1, 'τ_switch = 10⁻⁶', ha='center', va='center',
            fontsize=9, style='italic',
            bbox=dict(boxstyle="round,pad=0.25", facecolor='white',
                     edgecolor='orange', linewidth=1))
    
    # Add labels for sections
    ax.text(tr_x, tr_y-1.5, 'Computational Core', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0', style='italic')
    
    ax.text(guard_x, 1.5, 'Dual-Path Architecture', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1565C0', style='italic')
    
    # Add domain extension info (top left)
    ax.text(2, 6.5, 'Domain Extension:', fontsize=10, fontweight='bold')
    ax.text(2, 6.2, '𝕋 = ℝ ∪ {+∞, -∞, Φ}', fontsize=9)
    ax.text(2, 5.9, '• Total arithmetic', fontsize=8)
    ax.text(2, 5.6, '• No undefined ops', fontsize=8)
    
    # Add benefits (bottom right)
    ax.text(16, 2.2, 'Benefits:', fontsize=10, fontweight='bold')
    ax.text(16, 1.9, '• Bounded gradients', fontsize=8)
    ax.text(16, 1.6, '• Stable training', fontsize=8)
    ax.text(16, 1.3, '• Deterministic', fontsize=8)
    ax.text(16, 1.0, '• Unbiased', fontsize=8)
    
    # Add title at the TOP with proper spacing
    ax.text(9, 7.5, 'ZeroProofML: Transreal Computation Flow', 
            ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(9, 7.0, 'Singularities become well-defined computational states',
            ha='center', va='center', fontsize=11, style='italic', color='#555')
    
    # Set limits with space for title
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save with proper margins
    output_dir = Path('/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'figure6_perfect_schematic.png', 
                dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'figure6_perfect_schematic.pdf', 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✓ Perfect schematic created with proper title spacing and precise arrows!")
    return True

def create_ultra_clean_version():
    """Create an ultra-clean version with minimal elements"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Minimalist colors
    colors = {
        'process': '#F5F5F5',
        'guard': '#E8F5E9',
        'real': '#FFEBEE',
        'decision': '#FFF3E0'
    }
    
    # Define positions
    positions = {
        'input': (2, 3),
        'tr': (5, 3),
        'decision': (7.5, 3),
        'guard': (10, 4),
        'real': (10, 2),
        'output': (12.5, 3)
    }
    
    # Create boxes
    boxes = [
        ('input', 1.5, 0.6, 'Input\nx ∈ ℝⁿ', colors['process'], '#2196F3'),
        ('tr', 2, 0.6, 'P(x)/Q(x)', colors['process'], '#4CAF50'),
        ('guard', 1.8, 0.5, 'Guard: P/Q', colors['guard'], '#8BC34A'),
        ('real', 1.8, 0.5, 'Real: ∞/Φ', colors['real'], '#F44336'),
        ('output', 1.5, 0.6, 'Output\n(y, tag)', colors['process'], '#9C27B0')
    ]
    
    for key, width, height, text, fcolor, ecolor in boxes:
        x, y = positions[key]
        rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.05",
                             facecolor=fcolor, edgecolor=ecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10)
    
    # Decision point
    x, y = positions['decision']
    circle = Circle((x, y), 0.25, facecolor=colors['decision'], 
                   edgecolor='#FF9800', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, '?', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows
    arrows = [
        ('input', 'tr'),
        ('tr', 'decision'),
        ('decision', 'guard', '#4CAF50', 'YES'),
        ('decision', 'real', '#F44336', 'NO'),
        ('guard', 'output', '#8BC34A', ''),
        ('real', 'output', '#F44336', '')
    ]
    
    for i, arrow_info in enumerate(arrows):
        if len(arrow_info) == 2:
            start_key, end_key = arrow_info
            color = 'black'
            label = ''
        else:
            start_key, end_key, color, label = arrow_info
        
        start = positions[start_key]
        end = positions[end_key]
        
        if start_key == 'decision':
            # Adjust start position for decision arrows
            if end_key == 'guard':
                start = (start[0] + 0.2, start[1] + 0.2)
            else:
                start = (start[0] + 0.2, start[1] - 0.2)
        
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=color))
        
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.1, label, fontsize=8, color=color,
                   fontweight='bold', ha='center')
    
    # Title (with space above)
    ax.text(7, 5, 'ZeroProofML Architecture', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Condition label
    ax.text(
        positions['decision'][0],
        positions['decision'][1] - 0.6,
        r'$|Q|>\tau$',
        ha='center',
        va='center',
        fontsize=5,
    )
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save ultra-clean version
    output_dir = Path('/home/zsemed/ZeroProofML/results/robotics/paper_suite/figures')
    plt.savefig(output_dir / 'figure6_ultraclean.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'figure6_ultraclean.pdf', dpi=300, bbox_inches='tight')
    
    print("✓ Ultra-clean schematic created!")
    return True

if __name__ == '__main__':
    print("\nCreating perfect schematics with proper layout...")
    
    try:
        create_perfect_schematic()
    except Exception as e:
        print(f"Error creating perfect schematic: {e}")
    
    try:
        create_ultra_clean_version()
    except Exception as e:
        print(f"Error creating ultra-clean version: {e}")
    
    print("\n✓ Schematics complete!")
    print("  • figure6_perfect_schematic.pdf - Perfect layout with title at top")
    print("  • figure6_ultraclean.pdf - Ultra-minimal version")
