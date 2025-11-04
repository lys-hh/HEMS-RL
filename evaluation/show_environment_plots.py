
"""
Environment visualization plot summary
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_all_plots():
    """Display all saved plots"""
    plot_dir = 'save'
    
    if not os.path.exists(plot_dir):
        print("save folder does not exist!")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} images:")
    for i, filename in enumerate(sorted(image_files), 1):
        print(f"{i}. {filename}")
    
    # Create image display
    n_images = len(image_files)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(image_files):
        row = i // cols
        col = i % cols
        
        img_path = os.path.join(plot_dir, filename)
        img = mpimg.imread(img_path)
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(filename, fontsize=10)
        axes[row, col].axis('off')
    
    # Hide extra subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('figures/environment_plots/plot_summary.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    show_all_plots()
