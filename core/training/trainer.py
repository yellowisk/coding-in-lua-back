import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class XGBoostTrainer:
    def __init__(self, data_path='../data/processed-step-3.csv'):
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        df = pd.read_csv(self.data_path)
        
        # Assuming the last column is the target variable
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Split based on date: data until 2023 for training, 2024 for testing
            train_mask = df['date'] < '2024-01-01'
            test_mask = df['date'] >= '2024-01-01'
            
            self.X_train = X[train_mask].drop('date', axis=1)
            self.X_test = X[test_mask].drop('date', axis=1)
            self.y_train = y[train_mask]
            self.y_test = y[test_mask]
        else:
            raise ValueError("No 'date' column found in the dataset")
        
    def train(self, **kwargs):
        """Train the XGBoost classifier"""
        # Default parameters optimized for biological classification
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',  # Better for biological classification
            'tree_method': 'auto',  # Enable GPU acceleration
            'device': 'gpu',
            'random_state': 42,
            'n_estimators': 200
        }
        
        # Update with any provided parameters
        params.update(kwargs)
        
        # Initialize and train the model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Make predictions
        y_pred = self.model.predict(self.X_test)
                
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Generate classification report
        report = classification_report(self.y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Calculate training accuracy
        y_train_pred = self.model.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        return accuracy, report
        
    def save_model(self, model_path='models/xgboost_classifier.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
    def run_training_pipeline(self):
        """Complete training pipeline"""
        print("Loading data...")
        self.load_data()
        
        print("Training XGBoost classifier...")
        self.train()
        
        print("Evaluating model...")
        self.evaluate()
        
        print("Saving model...")
        self.save_model()

if __name__ == "__main__":
    trainer = XGBoostTrainer()
    trainer.run_training_pipeline()