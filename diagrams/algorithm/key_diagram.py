import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Define different style presets that can be easily switched
STYLE_PRESETS = {
    # Default style for general use
    "default": {
        # Figure settings
        "figure_width": 10,         # Total figure width in inches
        "figure_height": 4,         # Figure height in inches
        "dpi": 400,                 # DPI for saved figures
        "tight_layout_pad": 0.5,    # Padding around the figure
        "tight_layout_w_pad": 2.0,  # Width padding between subplots
        "tight_layout_h_pad": 0.5,  # Height padding between subplots
        
        # Title settings
        "title_fontsize": 12,       # Font size for subplot titles
        "title_pad": -5,            # Space between title and plot (negative brings closer)
        
        # Block settings
        "block_width": 1.5,         # Width of blocks
        "block_height": 0.6,        # Height of blocks
        "block_linewidth": 1.5,     # Line width of block borders
        "block_edgecolor": 'black', # Color of block borders
        "block_facecolor": 'none',  # Fill color of blocks (none = transparent)
        "block_fontsize": 10,       # Font size for text in blocks
        
        # Arrow settings
        "arrow_linewidth": 1.2,     # Line width for arrows
        "arrow_color": 'black',     # Color of arrows
        "arrow_style": '->',        # Arrow style
        
        # Text settings
        "label_fontsize": 10,       # Font size for labels
        "text_color": 'black',      # Color for text
        
        # Layout settings
        "x_min": -3,                # Minimum x-axis limit
        "x_max": 3.5,               # Maximum x-axis limit
        "y_min": -0.8,              # Minimum y-axis limit
        "y_max": 3.8,               # Maximum y-axis limit
    },
    
    # Style optimized for paper publication
    "paper": {
        # Figure settings
        "figure_width": 8,          # Narrower for paper columns
        "figure_height": 3.5,       # Shorter height
        "dpi": 600,                 # Higher DPI for print quality
        "tight_layout_pad": 0.1,    # Minimal padding to maximize content
        "tight_layout_w_pad": 0.5,  # Minimal width padding
        "tight_layout_h_pad": 0.1,  # Minimal height padding
        
        # Title settings
        "title_fontsize": 14,       # Slightly smaller font for titles
        "title_pad": -5,            # Bring title closer to content
        
        # Block settings
        "block_width": 1.4,         # Slightly narrower blocks
        "block_height": 0.55,       # Slightly shorter blocks
        "block_linewidth": 1.0,     # Thinner lines for print
        "block_edgecolor": 'black', # Standard black edges
        "block_facecolor": 'none',  # Transparent fill
        "block_fontsize": 12,        # Smaller font for blocks
        
        # Arrow settings
        "arrow_linewidth": 0.8,     # Thinner arrows for print
        "arrow_color": 'black',     # Standard black arrows
        "arrow_style": '->',        # Standard arrow style
        
        # Text settings
        "label_fontsize": 12,        # Smaller font for labels
        "text_color": 'black',      # Standard black text
        
        # Layout settings
        "x_min": -3,                # Same layout dimensions
        "x_max": 3.5,
        "y_min": -0.8,
        "y_max": 3.8,
    },
    
    # Style optimized for presentations
    "presentation": {
        # Figure settings
        "figure_width": 12,         # Wider for presentations
        "figure_height": 5,         # Taller for better visibility
        "dpi": 400,                 # Lower DPI is fine for screens
        "tight_layout_pad": 1.0,    # More padding for visibility
        "tight_layout_w_pad": 3.0,  # More space between subplots
        "tight_layout_h_pad": 1.0,  # More vertical space
        
        # Title settings
        "title_fontsize": 20,       # Larger font for visibility
        "title_pad": 5,             # More space between title and content
        
        # Block settings
        "block_width": 1.8,         # Wider blocks for visibility
        "block_height": 0.7,        # Taller blocks
        "block_linewidth": 2.0,     # Thicker lines for visibility
        "block_edgecolor": 'black', # Standard black edges
        "block_facecolor": 'none',  # Transparent fill
        "block_fontsize": 20,       # Larger font for visibility
        
        # Arrow settings
        "arrow_linewidth": 1.5,     # Thicker arrows for visibility
        "arrow_color": 'black',     # Standard black arrows
        "arrow_style": '->',        # Standard arrow style
        
        # Text settings
        "label_fontsize": 20,       # Larger font for visibility
        "text_color": 'black',      # Standard black text
        
        # Layout settings
        "x_min": -3.5,              # Slightly wider view
        "x_max": 4.0,
        "y_min": -1.0,
        "y_max": 4.0,
    }
}

# Select which style to use (change this to switch styles)
ACTIVE_STYLE = "paper"
style = STYLE_PRESETS[ACTIVE_STYLE]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Utility function: draw a rectangle (block) at a given center (x, y) with a label.
def draw_block(ax, center_x, center_y, label, width=None, height=None):
    # Use style parameters or provided values
    width = width if width is not None else style["block_width"]
    height = height if height is not None else style["block_height"]
    
    # Calculate lower-left corner
    ll_x = center_x - width/2
    ll_y = center_y - height/2
    
    # Create the rectangle patch
    rect = patches.Rectangle(
        (ll_x, ll_y), width, height,
        linewidth=style["block_linewidth"], 
        edgecolor=style["block_edgecolor"], 
        facecolor=style["block_facecolor"]
    )
    ax.add_patch(rect)
    
    # Add text in the center of the rectangle
    ax.text(
        center_x, center_y, label, 
        fontsize=style["block_fontsize"], 
        color=style["text_color"],
        ha='center', va='center'
    )

# Utility function: draw an arrow from (x1, y1) to (x2, y2) with optional text.
def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            facecolor=style["arrow_color"], 
            arrowstyle=style["arrow_style"], 
            lw=style["arrow_linewidth"]
        )
    )
    
    if text:
        mid_x = (x1 + x2)/2
        mid_y = (y1 + y2)/2
        ax.text(
            mid_x, mid_y, text, 
            fontsize=style["label_fontsize"], 
            color=style["text_color"],
            ha='center', va='bottom'
        )

def draw_method_a(ax):
    """
    Method (a): Each environment vector is transformed by a distinct orthonormal matrix
                U1, U2, U3, U4.
    """
    # Set title with style parameters
    ax.set_title('Random Projection', 
                fontsize=style["title_fontsize"], 
                pad=style["title_pad"])

    # Positions for 4 environment vectors (vertical stack)
    y_positions = [3, 2, 1, 0]  # top to bottom

    for i, y in enumerate(y_positions):
        # Input label
        env_label = f'Env{i+1}'
        ax.text(-2, y, env_label, 
                fontsize=style["label_fontsize"], 
                color=style["text_color"],
                ha='right', va='center')
        
        # Arrow from input to block
        draw_arrow(ax, -1.8, y, -0.9, y)
        
        # Block with label U_{i+1} with actual value of i
        u_label = rf'$U_{{{i+1}}}$'
        block_x = -0.2
        draw_block(ax, block_x, y, u_label)
        
        # Arrow from block to output
        draw_arrow(ax, block_x+0.75, y, block_x+1.8, y)
        
        # Output label
        ax.text(block_x+2.2, y, f'Input{i+1}', 
                fontsize=style["label_fontsize"], 
                color=style["text_color"],
                ha='left', va='center')

    # Adjust axis using style parameters
    ax.set_xlim([style["x_min"], style["x_max"]])
    ax.set_ylim([style["y_min"], style["y_max"]])
    ax.set_aspect('equal', 'box')
    ax.axis('off')

def draw_method_b(ax):
    """
    Method (b): Two orthonormal matrices U1, U2 are sampled, then we apply
                U1^2, U1U2, U2U1, and U2^2 respectively.
    """
    # Set title with style parameters
    ax.set_title(r'FRP($\ell=2$)', 
                fontsize=style["title_fontsize"], 
                pad=style["title_pad"])

    y_positions = [3, 2, 1, 0]  # top to bottom
    transform_labels = [
        r'$\lambda(aa)$', r'$\lambda(ab)$', r'$\lambda(ba)$', r'$\lambda(bb)$'
    ]

    for i, y in enumerate(y_positions):
        # Input label
        env_label = f'Env{i+1}'
        ax.text(-2, y, env_label, 
                fontsize=style["label_fontsize"], 
                color=style["text_color"],
                ha='right', va='center')
        
        # Arrow from input to block
        draw_arrow(ax, -1.8, y, -0.9, y)
        
        # Block
        block_x = -0.2
        draw_block(ax, block_x, y, transform_labels[i])
        
        # Arrow from block to output
        draw_arrow(ax, block_x+0.75, y, block_x+1.8, y)
        
        # Output label
        ax.text(block_x+2.2, y, f'Input{i+1}', 
                fontsize=style["label_fontsize"], 
                color=style["text_color"],
                ha='left', va='center')

    # Adjust axis using style parameters
    ax.set_xlim([style["x_min"], style["x_max"]])
    ax.set_ylim([style["y_min"], style["y_max"]])
    ax.set_aspect('equal', 'box')
    ax.axis('off')

def main():
    # Create figure with style parameters
    fig, axes = plt.subplots(
        1, 2, 
        figsize=(style["figure_width"], style["figure_height"])
    )

    # Draw each method in separate subplot
    draw_method_a(axes[0])
    draw_method_b(axes[1])

    # Use style parameters for layout
    plt.tight_layout(
        pad=style["tight_layout_pad"], 
        w_pad=style["tight_layout_w_pad"], 
        h_pad=style["tight_layout_h_pad"]
    )

    # Save figure as PDF and PNG with style parameters
    plt.savefig("key_diagram.pdf", dpi=style["dpi"], bbox_inches='tight')
    plt.savefig("key_diagram.png", bbox_inches='tight')

    # If you prefer to show the figure in a window:
    # plt.show()

if __name__ == '__main__':
    main()
