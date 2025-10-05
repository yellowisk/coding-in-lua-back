import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from itertools import combinations

class XGBoostHeatmapVisualizer:
    def __init__(self, model_path='../training/models/xgboost_classifier.pkl', 
                 data_path='../data/processed-step-3.csv'):
        """
        Initialize the visualizer with model and data paths
        
        Args:
            model_path: Path to the trained XGBoost model
            data_path: Path to the dataset
        """
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.feature_names = None
        
    def load_model(self):
        """Load the trained XGBoost model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")
        
    def load_data(self):
        """Load the dataset to understand feature ranges"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        
        # Extract feature names (excluding target and date if present)
        if 'shark' in self.data.columns:
            feature_cols = [col for col in self.data.columns if col not in ['shark', 'date']]
        else:
            feature_cols = [col for col in self.data.columns if col != 'date']
        
        self.feature_names = feature_cols
        print(f"Data loaded. Features: {self.feature_names}")
        
    def create_2d_heatmap(self, feature1, feature2, grid_size=50, figsize=(10, 8)):
        """
        Create a 2D heatmap showing model predictions for two features
        
        Args:
            feature1: Name of the first feature (x-axis)
            feature2: Name of the second feature (y-axis)
            grid_size: Number of points in each dimension
            figsize: Figure size tuple
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get feature ranges from data
        feature_data = self.data[self.feature_names]
        
        # Create meshgrid
        f1_min, f1_max = feature_data[feature1].min(), feature_data[feature1].max()
        f2_min, f2_max = feature_data[feature2].min(), feature_data[feature2].max()
        
        # Add some padding
        f1_range = f1_max - f1_min
        f2_range = f2_max - f2_min
        f1_min -= 0.1 * f1_range
        f1_max += 0.1 * f1_range
        f2_min -= 0.1 * f2_range
        f2_max += 0.1 * f2_range
        
        f1_grid = np.linspace(f1_min, f1_max, grid_size)
        f2_grid = np.linspace(f2_min, f2_max, grid_size)
        
        f1_mesh, f2_mesh = np.meshgrid(f1_grid, f2_grid)
        
        # Create prediction grid
        # Use median values for other features
        median_values = feature_data.median()
        
        predictions = np.zeros_like(f1_mesh)
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Create feature vector
                feature_vector = median_values.copy()
                feature_vector[feature1] = f1_mesh[i, j]
                feature_vector[feature2] = f2_mesh[i, j]
                
                # Reshape for prediction
                X = feature_vector.values.reshape(1, -1)
                
                # Get prediction probability
                pred_proba = self.model.predict_proba(X)[0, 1]
                predictions[i, j] = pred_proba
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.contourf(f1_mesh, f2_mesh, predictions, levels=20, cmap='RdYlGn_r')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Shark Presence Probability', rotation=270, labelpad=20)
        
        # Add contour lines
        contour = ax.contour(f1_mesh, f2_mesh, predictions, levels=[0.5], 
                            colors='black', linewidths=2, linestyles='--')
        ax.clabel(contour, inline=True, fontsize=10)
        
        # Scatter actual data points
        actual_data = self.data[[feature1, feature2, 'shark']].dropna()
        sharks = actual_data[actual_data['shark'] == 1]
        # no_sharks = actual_data[actual_data['shark'] == 0]
        
        # ax.scatter(no_sharks[feature1], no_sharks[feature2], 
        #           c='blue', marker='o', s=20, alpha=0.3, label='No Shark')
        ax.scatter(sharks[feature1], sharks[feature2], 
                  c='red', marker='^', s=30, alpha=0.6, label='Shark Present')
        
        ax.set_xlabel(feature1.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(feature2.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Shark Presence Probability: {feature1} vs {feature2}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, ax
    
    def create_multiple_heatmaps(self, feature_pairs, grid_size=50, save_dir='heatmaps'):
        """
        Create multiple heatmaps for different feature pairs
        
        Args:
            feature_pairs: List of tuples containing feature pairs
            grid_size: Number of points in each dimension
            save_dir: Directory to save the plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for feature1, feature2 in feature_pairs:
            if feature1 not in self.feature_names or feature2 not in self.feature_names:
                print(f"Warning: Features {feature1} or {feature2} not found in data. Skipping.")
                continue
            
            print(f"Creating heatmap for {feature1} vs {feature2}...")
            
            fig, ax = self.create_2d_heatmap(feature1, feature2, grid_size)
            
            # Save the figure
            filename = f"{feature1}_vs_{feature2}_heatmap_only_presence.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            
            plt.close(fig)
    
    def create_all_important_combinations(self, top_features=None, grid_size=50, save_dir='heatmaps'):
        """
        Create heatmaps for all important feature combinations
        
        Args:
            top_features: List of important features. If None, uses all features.
            grid_size: Number of points in each dimension
            save_dir: Directory to save the plots
        """
        if top_features is None:
            top_features = self.feature_names
        
        # Generate all combinations of features
        feature_combinations = list(combinations(top_features, 2))
        
        print(f"Creating {len(feature_combinations)} heatmaps...")
        self.create_multiple_heatmaps(feature_combinations, grid_size, save_dir)
    
    def plot_feature_importance(self, figsize=(10, 6)):
        """Plot feature importance from the XGBoost model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        plt.tight_layout()
        
        return fig, ax, feature_importance


def main():
    """Main function to run the visualization pipeline"""
    # Initialize visualizer
    visualizer = XGBoostHeatmapVisualizer(
        model_path='../training/models/xgboost_classifier.pkl',
        data_path='../data/processed-step-3.csv'
    )
    
    # Load model and data
    print("Loading model and data...")
    visualizer.load_model()
    visualizer.load_data()
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    fig_imp, ax_imp, importance_df = visualizer.plot_feature_importance()
    os.makedirs('heatmaps', exist_ok=True)
    fig_imp.savefig('heatmaps/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Feature importance saved to heatmaps/feature_importance.png")
    print("\nTop features:")
    print(importance_df)
    plt.close(fig_imp)
    
    # Define specific feature pairs of interest
    feature_pairs = [
        ('temperature', 'ocean_depth'),
        ('phyto', 'temperature'),
        ('phyto', 'ocean_depth'),
        ('temperature', 'max_individual_wave_height'),
        ('ocean_depth', 'max_individual_wave_height'),
        ('phyto', 'max_individual_wave_height'),
        ('temperature', 'clouds'),
        ('phyto', 'clouds'),
        ('lon', 'lat'),
        ('mean_wave_direction', 'mean_period_total_swell'),
    ]
    
    # Create heatmaps for specified pairs
    print("\nCreating 2D heatmaps...")
    visualizer.create_multiple_heatmaps(feature_pairs, grid_size=50, save_dir='heatmaps')
    
    print("\nAll visualizations complete! Check the 'heatmaps' directory.")


if __name__ == '__main__':
    main()
