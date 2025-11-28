# visualize_architecture.py
"""
Model architecture visualization in publication style
Similar to Transformer, ResNet, U-Net papers
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches

# Publication-quality settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2


def draw_block(ax, x, y, w, h, text, color, fontsize=20, subtext=''):
    """Draw a simple rectangular block with text"""
    rect = Rectangle((x - w/2, y - h/2), w, h,
                     facecolor=color, edgecolor='black',
                     linewidth=3, zorder=2)
    ax.add_patch(rect)

    ax.text(x, y + 0.15, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=3)

    if subtext:
        ax.text(x, y - 0.2, subtext, ha='center', va='center',
                fontsize=fontsize-4, color='white', zorder=3, style='italic')


def draw_arrow(ax, x1, y1, x2, y2, width=0.08):
    """Draw a thick arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=40,
                           linewidth=3, color='black', zorder=1)
    ax.add_patch(arrow)


def visualize_baseline():
    """Baseline CNN - Simple vertical stack"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 15)
    ax.axis('off')

    # Title
    ax.text(0, 14.5, 'Baseline CNN Architecture', ha='center',
            fontsize=28, fontweight='bold')
    ax.text(0, 13.8, '(Khait et al., 2023)', ha='center',
            fontsize=20, style='italic', color='gray')

    y = 12.5

    # Input
    draw_block(ax, 0, y, 2.5, 0.8, 'Input', '#95A5A6', 18, '[B, 1, 1000]')
    draw_arrow(ax, 0, y - 0.4, 0, y - 1.1)
    y -= 1.8

    # Conv Block 1
    draw_block(ax, 0, y, 3, 1.2, 'Conv Block 1', '#3498DB', 20,
               '2×Conv1D(32, k=9)')
    draw_arrow(ax, 0, y - 0.6, 0, y - 1.3)
    y -= 1.6

    draw_block(ax, 0, y, 2.5, 0.7, 'MaxPool(4)', '#E74C3C', 16)
    draw_arrow(ax, 0, y - 0.35, 0, y - 0.85)
    y -= 1.1

    draw_block(ax, 0, y, 2.5, 0.6, 'Dropout(0.5)', '#7F8C8D', 14)
    draw_arrow(ax, 0, y - 0.3, 0, y - 0.8)
    y -= 1.1

    # Conv Block 2
    draw_block(ax, 0, y, 3, 1.2, 'Conv Block 2', '#2980B9', 20,
               '2×Conv1D(64, k=9)')
    draw_arrow(ax, 0, y - 0.6, 0, y - 1.3)
    y -= 1.6

    draw_block(ax, 0, y, 2.5, 0.7, 'MaxPool(4)', '#E74C3C', 16)
    draw_arrow(ax, 0, y - 0.35, 0, y - 0.85)
    y -= 1.1

    draw_block(ax, 0, y, 2.5, 0.6, 'Dropout(0.5)', '#7F8C8D', 14)
    draw_arrow(ax, 0, y - 0.3, 0, y - 0.8)
    y -= 1.1

    # Conv Block 3
    draw_block(ax, 0, y, 3, 1.2, 'Conv Block 3', '#1ABC9C', 20,
               '2×Conv1D(128, k=9)')
    draw_arrow(ax, 0, y - 0.6, 0, y - 1.3)
    y -= 1.6

    draw_block(ax, 0, y, 2.5, 0.7, 'MaxPool(4)', '#E74C3C', 16)
    draw_arrow(ax, 0, y - 0.35, 0, y - 0.85)
    y -= 1.1

    draw_block(ax, 0, y, 2.5, 0.6, 'Dropout(0.5)', '#7F8C8D', 14)
    draw_arrow(ax, 0, y - 0.3, 0, y - 0.8)
    y -= 1.1

    # Flatten + Dense
    draw_block(ax, 0, y, 2.5, 0.7, 'Flatten', '#F39C12', 16)
    draw_arrow(ax, 0, y - 0.35, 0, y - 0.85)
    y -= 1.1

    draw_block(ax, 0, y, 3, 0.9, 'Dense(128)', '#9B59B6', 18, 'ReLU + Dropout')
    draw_arrow(ax, 0, y - 0.45, 0, y - 1.0)
    y -= 1.3

    # Output
    draw_block(ax, 0, y, 2.5, 0.8, 'Output', '#27AE60', 18, 'Binary Logit')

    plt.tight_layout()
    plt.savefig('experiments/figures/baseline_cnn_architecture.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: baseline_cnn_architecture.png")
    plt.close()


def visualize_mymodel():
    """MyModel - Backbone + 3 branches"""
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.set_xlim(-5, 13)
    ax.set_ylim(-1, 17)
    ax.axis('off')

    # Title
    ax.text(4, 16.5, 'MyModel Architecture', ha='center',
            fontsize=32, fontweight='bold')
    ax.text(4, 15.7, '(Baseline CNN + VAE + SSL + DG)', ha='center',
            fontsize=22, style='italic', color='gray')

    y = 14

    # Input
    draw_block(ax, 4, y, 2.5, 0.8, 'Input', '#95A5A6', 18, '[B, 1, 1000]')
    draw_arrow(ax, 4, y - 0.4, 4, y - 1.1)
    y -= 1.8

    # ========== CNN Backbone (simplified) ==========
    # Draw background box for backbone
    backbone_box = FancyBboxPatch((1.5, y - 4.5), 5, 5.2,
                                  boxstyle="round,pad=0.15",
                                  facecolor='#ECF0F1', edgecolor='#2C3E50',
                                  linewidth=4, alpha=0.3, zorder=0)
    ax.add_patch(backbone_box)
    ax.text(4, y + 0.8, 'CNN Backbone', ha='center',
            fontsize=22, fontweight='bold', color='#2C3E50')
    ax.text(4, y + 0.3, '(Same as Baseline)', ha='center',
            fontsize=16, style='italic', color='#34495E')

    # Conv blocks (simplified)
    draw_block(ax, 4, y, 3.5, 1.0, 'Conv Block 1', '#3498DB', 18,
               '32 filters → Pool → Drop')
    draw_arrow(ax, 4, y - 0.5, 4, y - 1.2)
    y -= 1.5

    draw_block(ax, 4, y, 3.5, 1.0, 'Conv Block 2', '#2980B9', 18,
               '64 filters → Pool → Drop')
    draw_arrow(ax, 4, y - 0.5, 4, y - 1.2)
    y -= 1.5

    draw_block(ax, 4, y, 3.5, 1.0, 'Conv Block 3', '#1ABC9C', 18,
               '128 filters → Pool → Drop')
    draw_arrow(ax, 4, y - 0.5, 4, y - 1.2)
    y -= 1.5

    draw_block(ax, 4, y, 3.5, 0.8, 'Flatten + Dense(128)', '#F39C12', 16)
    draw_arrow(ax, 4, y - 0.4, 4, y - 1.0)
    y -= 1.3

    # ========== Embedding h ==========
    draw_block(ax, 4, y, 4, 1.0, 'Embedding h', '#8E44AD', 22, '[B, 128]')

    # Three arrows going down to branches
    branch_y = y - 1.5
    draw_arrow(ax, 4, y - 0.5, 1, branch_y + 0.5)
    draw_arrow(ax, 4, y - 0.5, 4, branch_y + 0.5)
    draw_arrow(ax, 4, y - 0.5, 10, branch_y + 0.5)

    # ========== Branch 1: Main Classifier ==========
    bx1 = 1
    by = branch_y

    ax.text(bx1, by + 0.8, 'Main Classifier', ha='center',
            fontsize=20, fontweight='bold', color='#27AE60')

    draw_block(ax, bx1, by, 2.5, 0.8, 'Dense(1)', '#27AE60', 16)
    draw_arrow(ax, bx1, by - 0.4, bx1, by - 1.0)
    by -= 1.3

    draw_block(ax, bx1, by, 2.5, 0.8, 'Binary Logit', '#27AE60', 16)
    by -= 1.2

    ax.text(bx1, by, 'BCE Loss', ha='center',
            fontsize=14, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#D5F4E6',
                     edgecolor='#27AE60', linewidth=2))

    # ========== Branch 2: VAE ==========
    bx2 = 4
    by = branch_y

    ax.text(bx2, by + 0.8, 'VAE Module', ha='center',
            fontsize=20, fontweight='bold', color='#E74C3C')

    # Encoder - two parallel paths
    draw_block(ax, bx2 - 0.8, by, 1.3, 0.7, 'μ', '#E67E22', 14, '[B, 32]')
    draw_block(ax, bx2 + 0.8, by, 1.3, 0.7, 'log σ²', '#E67E22', 14, '[B, 32]')

    draw_arrow(ax, bx2 - 0.8, by - 0.35, bx2, by - 1.1)
    draw_arrow(ax, bx2 + 0.8, by - 0.35, bx2, by - 1.1)
    by -= 1.4

    # Reparameterization
    draw_block(ax, bx2, by, 2.8, 0.7, 'z = μ + σ⊙ε', '#E74C3C', 14)
    draw_arrow(ax, bx2, by - 0.35, bx2, by - 0.9)
    by -= 1.2

    # Latent z
    draw_block(ax, bx2, by, 2.5, 0.8, 'Latent z', '#C0392B', 16, '[B, 32]')
    draw_arrow(ax, bx2, by - 0.4, bx2, by - 1.0)
    by -= 1.3

    # Decoder
    draw_block(ax, bx2, by, 2.5, 0.7, 'Decoder', '#D35400', 14, 'Dense(128)')
    draw_arrow(ax, bx2, by - 0.35, bx2, by - 0.9)
    by -= 1.2

    draw_block(ax, bx2, by, 2.5, 0.7, 'h_rec', '#E74C3C', 14, '[B, 128]')
    by -= 1.0

    ax.text(bx2, by, 'MSE + β·KL', ha='center',
            fontsize=14, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8',
                     edgecolor='#E74C3C', linewidth=2))

    # SSL annotation
    ax.annotate('', xy=(bx2 + 1.8, branch_y - 1.5), xytext=(bx2 - 1.8, branch_y - 1.5),
                arrowprops=dict(arrowstyle='<->', lw=3, color='#F39C12'))
    ax.text(bx2, branch_y - 2.0, 'SSL: z ≈ z_aug', ha='center',
            fontsize=14, fontweight='bold', color='#F39C12',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF5E7',
                     edgecolor='#F39C12', linewidth=2))

    # ========== Branch 3: DG ==========
    bx3 = 10
    by = branch_y

    ax.text(bx3, by + 0.8, 'Domain Generalization', ha='center',
            fontsize=20, fontweight='bold', color='#3498DB')

    draw_block(ax, bx3, by, 2.8, 0.8, 'Domain Head', '#3498DB', 16)
    draw_arrow(ax, bx3, by - 0.4, bx3, by - 1.0)
    by -= 1.3

    draw_block(ax, bx3, by, 2.8, 0.8, 'Domain Logits', '#3498DB', 16, '[B, 2]')
    by -= 1.2

    ax.text(bx3, by, 'CE Loss', ha='center',
            fontsize=14, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#D6EAF8',
                     edgecolor='#3498DB', linewidth=2))

    plt.tight_layout()
    plt.savefig('experiments/figures/mymodel_architecture.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: mymodel_architecture.png")
    plt.close()


def visualize_comparison():
    """Side-by-side comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

    for ax in [ax1, ax2]:
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 13)
        ax.axis('off')

    # ===== LEFT: Baseline =====
    ax1.text(2, 12.5, 'Baseline CNN', ha='center',
             fontsize=28, fontweight='bold')

    y = 11
    draw_block(ax1, 2, y, 2.5, 0.7, 'Input', '#95A5A6', 16, '[B, 1, 1000]')
    draw_arrow(ax1, 2, y - 0.35, 2, y - 0.9)
    y -= 1.2

    # Backbone box
    backbone_box = FancyBboxPatch((0, y - 4.5), 4, 5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ECF0F1', edgecolor='#2C3E50',
                                  linewidth=3, alpha=0.3, zorder=0)
    ax1.add_patch(backbone_box)
    ax1.text(2, y + 0.5, 'CNN Backbone', ha='center',
             fontsize=18, fontweight='bold', color='#2C3E50')

    draw_block(ax1, 2, y, 3, 0.9, 'Conv × 3 Blocks', '#3498DB', 16,
               '32→64→128')
    draw_arrow(ax1, 2, y - 0.45, 2, y - 1.1)
    y -= 1.4

    draw_block(ax1, 2, y, 2.5, 0.7, 'Flatten+Dense', '#F39C12', 14, '128-dim')
    draw_arrow(ax1, 2, y - 0.35, 2, y - 1.0)
    y -= 1.3

    draw_block(ax1, 2, y, 2.5, 0.8, 'Output', '#27AE60', 16, 'Binary')

    ax1.text(2, 2, 'Single Task\nClassification', ha='center',
             fontsize=16, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#D5F4E6',
                      edgecolor='#27AE60', linewidth=2))

    # ===== RIGHT: MyModel =====
    ax2.text(2, 12.5, 'MyModel (Enhanced)', ha='center',
             fontsize=28, fontweight='bold')

    y = 11
    draw_block(ax2, 2, y, 2.5, 0.7, 'Input', '#95A5A6', 16, '[B, 1, 1000]')
    draw_arrow(ax2, 2, y - 0.35, 2, y - 0.9)
    y -= 1.2

    # Backbone box
    backbone_box = FancyBboxPatch((0, y - 4.5), 4, 5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#ECF0F1', edgecolor='#2C3E50',
                                  linewidth=3, alpha=0.3, zorder=0)
    ax2.add_patch(backbone_box)
    ax2.text(2, y + 0.5, 'Same CNN Backbone', ha='center',
             fontsize=18, fontweight='bold', color='#2C3E50')

    draw_block(ax2, 2, y, 3, 0.9, 'Conv × 3 Blocks', '#3498DB', 16,
               '32→64→128')
    draw_arrow(ax2, 2, y - 0.45, 2, y - 1.1)
    y -= 1.4

    draw_block(ax2, 2, y, 2.5, 0.7, 'Flatten+Dense', '#F39C12', 14, '128-dim')
    draw_arrow(ax2, 2, y - 0.35, 2, y - 0.9)
    y -= 1.2

    # Embedding
    draw_block(ax2, 2, y, 3, 0.8, 'Embedding h', '#8E44AD', 16, '[B, 128]')

    # Three branches
    by = y - 1.5
    draw_arrow(ax2, 2, y - 0.4, 0.5, by + 0.4)
    draw_arrow(ax2, 2, y - 0.4, 2, by + 0.4)
    draw_arrow(ax2, 2, y - 0.4, 3.5, by + 0.4)

    draw_block(ax2, 0.5, by, 1, 0.6, 'Main', '#27AE60', 12)
    draw_block(ax2, 2, by, 1, 0.6, 'VAE', '#E74C3C', 12)
    draw_block(ax2, 3.5, by, 1, 0.6, 'DG', '#3498DB', 12)

    ax2.text(2, 1.5, 'Multi-Task Learning:', ha='center',
             fontsize=16, fontweight='bold')
    ax2.text(2, 1.0, '• Classification\n• VAE (Representation)\n• SSL (Semi-Supervised)\n• DG (Domain Robust)',
             ha='center', fontsize=14,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1',
                      edgecolor='#2C3E50', linewidth=2))

    plt.tight_layout()
    plt.savefig('experiments/figures/architecture_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: architecture_comparison.png")
    plt.close()


def main():
    print("="*60)
    print("Generating Publication-Style Architecture Diagrams")
    print("="*60)

    print("\n1. Baseline CNN Architecture...")
    visualize_baseline()

    print("\n2. MyModel Architecture...")
    visualize_mymodel()

    print("\n3. Comparison Diagram...")
    visualize_comparison()

    print("\n" + "="*60)
    print("✓ All diagrams generated successfully!")
    print("="*60)
    print("\nFiles saved in: experiments/figures/")
    print("  • baseline_cnn_architecture.png")
    print("  • mymodel_architecture.png")
    print("  • architecture_comparison.png")


if __name__ == "__main__":
    main()
