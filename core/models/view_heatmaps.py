"""
Simple script to view the generated heatmaps interactively
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_heatmaps(heatmap_dir='heatmaps'):
    """Display all generated heatmaps"""
    
    if not os.path.exists(heatmap_dir):
        print(f"Directory {heatmap_dir} not found!")
        return
    
    # Get all PNG files
    heatmap_files = [f for f in os.listdir(heatmap_dir) if f.endswith('.png')]
    heatmap_files.sort()
    
    print(f"Found {len(heatmap_files)} heatmaps:")
    for i, filename in enumerate(heatmap_files, 1):
        print(f"{i}. {filename}")
    
    # Display each heatmap
    for filename in heatmap_files:
        filepath = os.path.join(heatmap_dir, filename)
        
        # Load and display image
        img = mpimg.imread(filepath)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(filename.replace('_', ' ').replace('.png', '').title(), 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user input to continue
        response = input("\nPress Enter to see next heatmap, or 'q' to quit: ")
        plt.close(fig)
        
        if response.lower() == 'q':
            break
    
    print("\nDone viewing heatmaps!")


def display_single_heatmap(filename, heatmap_dir='heatmaps'):
    """Display a single heatmap"""
    filepath = os.path.join(heatmap_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"File {filepath} not found!")
        return
    
    img = mpimg.imread(filepath)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(filename.replace('_', ' ').replace('.png', '').title(), 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Display specific heatmap
        display_single_heatmap(sys.argv[1])
    else:
        # Display all heatmaps
        display_heatmaps()
