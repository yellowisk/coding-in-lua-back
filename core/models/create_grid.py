"""
Create a grid of heatmaps for easy comparison
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def create_heatmap_grid(heatmap_dir='heatmaps', output_file='heatmap_summary.png'):
    """Create a grid showing multiple heatmaps"""
    
    # Specific heatmaps to include in the grid
    selected_heatmaps = [
        'feature_importance.png',
        'temperature_vs_ocean_depth_heatmap.png',
        'phyto_vs_temperature_heatmap.png',
        'phyto_vs_ocean_depth_heatmap.png',
        'lat_vs_lon_heatmap.png',
        'ocean_depth_vs_max_individual_wave_height_heatmap.png',
    ]
    
    # Filter to only existing files
    existing_files = [f for f in selected_heatmaps 
                     if os.path.exists(os.path.join(heatmap_dir, f))]
    
    if not existing_files:
        print("No heatmap files found!")
        return
    
    # Create grid layout
    n_plots = len(existing_files)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    
    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    # Load and display each heatmap
    for idx, filename in enumerate(existing_files):
        filepath = os.path.join(heatmap_dir, filename)
        img = mpimg.imread(filepath)
        
        ax = axes_flat[idx]
        ax.imshow(img)
        ax.axis('off')
        
        # Create title from filename
        title = filename.replace('_heatmap.png', '').replace('_', ' ').title()
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Hide unused subplots
    for idx in range(len(existing_files), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Shark Presence Probability Heatmaps - Overview', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save the grid
    output_path = os.path.join(heatmap_dir, output_file)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Heatmap grid saved to: {output_path}")
    
    plt.show()


def create_comparison_grid(feature_pairs, heatmap_dir='heatmaps', 
                          output_file='comparison_grid.png'):
    """Create a grid comparing specific feature pairs"""
    
    filenames = [f"{f1}_vs_{f2}_heatmap.png" for f1, f2 in feature_pairs]
    
    # Filter to only existing files
    existing_files = [(f, p) for f, p in zip(filenames, feature_pairs) 
                     if os.path.exists(os.path.join(heatmap_dir, f))]
    
    if not existing_files:
        print("No matching heatmap files found!")
        return
    
    n_plots = len(existing_files)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for idx, (filename, (f1, f2)) in enumerate(existing_files):
        filepath = os.path.join(heatmap_dir, filename)
        img = mpimg.imread(filepath)
        
        ax = axes_flat[idx]
        ax.imshow(img)
        ax.axis('off')
        
        title = f"{f1.replace('_', ' ').title()} vs {f2.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    for idx in range(len(existing_files), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Feature Combination Comparison', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(heatmap_dir, output_file)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Comparison grid saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # Create main overview grid
    print("Creating overview grid...")
    create_heatmap_grid()
    
    # Create comparison grid for key feature pairs
    print("\nCreating comparison grid...")
    key_pairs = [
        ('temperature', 'ocean_depth'),
        ('phyto', 'temperature'),
        ('phyto', 'ocean_depth'),
        ('ocean_depth', 'max_individual_wave_height'),
    ]
    create_comparison_grid(key_pairs, output_file='key_features_comparison.png')
    
    print("\nDone! Check the heatmaps directory.")
