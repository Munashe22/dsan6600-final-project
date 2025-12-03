"""
Model Training for Traffic Congestion Prediction

This module trains machine learning models on extracted video features to predict
congestion levels. No backpropagation during inference - using traditional ML.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class CongestionPredictor:
    """
    Traffic congestion prediction model using traditional ML
    No backpropagation during inference - feature extraction + traditional classifier
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictor
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: Optional[str] = None,
                        fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training or prediction
        
        Args:
            df: DataFrame with features
            target_col: Name of target column (None for prediction)
            fit_scaler: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (X, y) where y is None for prediction
        """
        # Exclude non-feature columns
        exclude_cols = ['segment_id', 'congestion_enter_rating', 'congestion_exit_rating']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        # Handle any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X = self.scaler.transform(X)
        
        # Prepare target if provided
        y = None
        if target_col and target_col in df.columns:
            if fit_scaler:
                y = self.label_encoder.fit_transform(df[target_col])
            else:
                y = self.label_encoder.transform(df[target_col])
        
        return X, y
    
    def train(self, 
             X: np.ndarray, 
             y: np.ndarray,
             validation_split: float = 0.2) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training {self.model_type} on {len(X_train)} samples...")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        from sklearn.metrics import f1_score
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        train_f1_weighted = f1_score(y_train, train_pred, average='weighted')
        val_f1_weighted = f1_score(y_val, val_pred, average='weighted')
        
        train_f1_macro = f1_score(y_train, train_pred, average='macro')
        val_f1_macro = f1_score(y_val, val_pred, average='macro')
        
        print(f"\n{'='*60}")
        print("TRAINING METRICS")
        print(f"{'='*60}")
        print(f"Training:")
        print(f"  Accuracy:           {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  F1 Score (Weighted): {train_f1_weighted:.4f}")
        print(f"  F1 Score (Macro):    {train_f1_macro:.4f}")
        print(f"\nValidation:")
        print(f"  Accuracy:           {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  F1 Score (Weighted): {val_f1_weighted:.4f}")
        print(f"  F1 Score (Macro):    {val_f1_macro:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            
            # Get top features
            if self.feature_names:
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.feature_importance
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(importance_df.head(10).to_string(index=False))
        
        # Classification report
        print(f"\n{'='*60}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(classification_report(
            y_val, val_pred,
            target_names=self.label_encoder.classes_
        ))
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1_weighted': train_f1_weighted,
            'val_f1_weighted': val_f1_weighted,
            'train_f1_macro': train_f1_macro,
            'val_f1_macro': val_f1_macro,
            'model_type': self.model_type,
            'n_features': X.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted class labels
        """
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Generate feature importance report for documentation
        
        Returns:
            DataFrame with feature importance information
        """
        if self.feature_importance is None or self.feature_names is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature Name': self.feature_names,
            'Importance Score': self.feature_importance,
            'Normalized Importance': self.feature_importance / self.feature_importance.sum()
        }).sort_values('Importance Score', ascending=False)
        
        # Add interpretable notes
        notes = []
        for feature in importance_df['Feature Name']:
            if 'occupancy' in feature.lower():
                note = "Measures road space occupied by vehicles - key congestion indicator"
            elif 'vehicle_count' in feature.lower():
                note = "Number of vehicles detected - direct traffic volume measure"
            elif 'motion' in feature.lower():
                note = "Vehicle movement patterns - distinguishes flowing vs stopped traffic"
            elif 'queue' in feature.lower():
                note = "Estimated queue length - critical for delay classification"
            elif 'speed' in feature.lower():
                note = "Average speed estimate - lower speeds indicate congestion"
            elif 'flow_rate' in feature.lower():
                note = "Vehicles per time unit - throughput measure"
            elif 'stop_frequency' in feature.lower():
                note = "Stop-and-go pattern frequency - indicates traffic interruptions"
            elif 'rolling' in feature.lower():
                note = "Temporal trend feature - captures congestion buildup over time"
            elif 'trend' in feature.lower():
                note = "Change pattern - detects increasing/decreasing congestion"
            else:
                note = "Additional traffic pattern indicator"
            
            notes.append(note)
        
        importance_df['Notes'] = notes
        
        return importance_df


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust predictions
    """
    
    def __init__(self, n_models: int = 3):
        """
        Initialize ensemble
        
        Args:
            n_models: Number of models in ensemble
        """
        self.models = [
            CongestionPredictor('random_forest'),
            CongestionPredictor('gradient_boosting'),
            CongestionPredictor('random_forest')  # Second RF with different seed
        ][:n_models]
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train all models in ensemble"""
        results = []
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*50}")
            print(f"Training Model {i+1}/{len(self.models)}")
            print(f"{'='*50}")
            result = model.train(X, y)
            results.append(result)
        
        return {
            'individual_results': results,
            'avg_val_accuracy': np.mean([r['val_accuracy'] for r in results])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ensemble voting
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted labels
        """
        # Get predictions from all models
        predictions = [model.predict(X) for model in self.models]
        
        # Majority voting
        predictions_array = np.array(predictions)
        
        # For each sample, take the most common prediction
        final_predictions = []
        for i in range(X.shape[0]):
            sample_preds = predictions_array[:, i]
            unique, counts = np.unique(sample_preds, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)


def train_pipeline(train_features_df: pd.DataFrame,
                   target_col: str,
                   model_save_path: str,
                   model_type: str = 'random_forest') -> CongestionPredictor:
    """
    Complete training pipeline
    
    Args:
        train_features_df: DataFrame with extracted features and labels
        target_col: Name of target column
        model_save_path: Path to save trained model
        model_type: Type of model to train
        
    Returns:
        Trained predictor
    """
    print(f"Training pipeline for {target_col}")
    print(f"Dataset shape: {train_features_df.shape}")
    print(f"Target distribution:\n{train_features_df[target_col].value_counts()}")
    
    # Initialize predictor
    predictor = CongestionPredictor(model_type=model_type)
    
    # Prepare features
    X, y = predictor.prepare_features(
        train_features_df,
        target_col=target_col,
        fit_scaler=True
    )
    
    # Train
    metrics = predictor.train(X, y)
    
    # Save model
    predictor.save(model_save_path)
    
    return predictor


if __name__ == "__main__":
    print("Traffic Congestion Model Training Module")
    print("="*50)
    print("\nThis module trains traditional ML models on video features")
    print("No backpropagation during inference - using Random Forest / Gradient Boosting")
    print("\nKey Features:")
    print("- Feature scaling and preprocessing")
    print("- Model training with validation")
    print("- Feature importance analysis")
    print("- Ensemble predictions")