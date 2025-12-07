"""
ANN Model Training for Traffic Congestion Prediction

This module trains a simple feedforward neural network on extracted video features.
Uses PyTorch, no backpropagation during inference, just forward pass.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb 
warnings.filterwarnings('ignore')


class CongestionANN(nn.Module):
    """
    Feedforward neural network for congestion classification
    Architecture: Input -> Hidden layers with ReLU + Dropout -> Output
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_classes: int = 3,
                 dropout: float = 0.3):
        """
        Initialize ANN
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(CongestionANN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class ANNPredictor:
    """
    Traffic congestion prediction using ANN
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.2,
                 learning_rate: float = 0.0001,
                 batch_size: int = 32,
                 epochs: int = 150,
                 early_stopping_patience: int = 10):
        """
        Initialize ANN predictor
        
        Args:
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.num_classes = None
        
    def prepare_features(self, df: pd.DataFrame,
                        target_col: Optional[str] = None,
                        fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      
        # remove non-feature columns
        remove_cols = ['segment_id', 'congestion_enter_rating', 'congestion_exit_rating']
        feature_cols = [col for col in df.columns if col not in remove_cols]
        
        X = df[feature_cols].values
        
        # handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # scale
        if fit_scaler:
            X = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X = self.scaler.transform(X)
        
        # prep target
        y = None
        if target_col and target_col in df.columns:
            if fit_scaler:
                y = self.label_encoder.fit_transform(df[target_col])
                self.num_classes = len(self.label_encoder.classes_)
            else:
                y = self.label_encoder.transform(df[target_col])
        
        return X, y
    
    def ann_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch dataloaders"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _save_confusion_matrix_and_metrics(self, y_true, y_pred, prefix, output_dir='models'):
        """Generate and save confusion matrix plot and metrics CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get class labels
        labels = self.label_encoder.classes_
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Create metrics DataFrame
        metrics_data = {
            'metric': ['Accuracy', 
                      'Precision (Macro)', 'Precision (Weighted)',
                      'Recall (Macro)', 'Recall (Weighted)',
                      'F1-Score (Macro)', 'F1-Score (Weighted)'],
            'value': [accuracy,
                     precision_macro, precision_weighted,
                     recall_macro, recall_weighted,
                     f1_macro, f1_weighted]
        }
        
        # Add per-class metrics
        for i, label in enumerate(labels):
            metrics_data['metric'].extend([
                f'Precision - {label}',
                f'Recall - {label}',
                f'F1-Score - {label}'
            ])
            metrics_data['value'].extend([
                precision_per_class[i],
                recall_per_class[i],
                f1_per_class[i]
            ])
        
        # Save metrics CSV
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = output_dir / f'{prefix}_metrics.csv'
        metrics_df.to_csv(csv_path, index=False)
        print(f"  Saved metrics: {csv_path}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        # Add percentages
        cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        for i in range(len(labels)):
            for j in range(len(labels)):
                percentage = cm_percentages[i, j]
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=9, color='gray')
        
        plt.title(f'Confusion Matrix - {prefix}\nAccuracy: {accuracy:.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'{prefix}_confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved confusion matrix: {plot_path}")
    
    def train(self,
             X: np.ndarray,
             y: np.ndarray,
             validation_split: float = 0.2,
             model_name: str = 'model') -> Dict:
        """
        Train ANN
        
        Args:
            X: Feature matrix
            y: Target labels
            validation_split: Fraction of data for validation
            model_name: Prefix for saved confusion matrices and metrics
            
        Returns:
            Dictionary with training metrics
        """
        # split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training ANN on {len(X_train)} samples.")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Device: {self.device}")
        
        # initialize model
        input_dim = X.shape[1]
        self.model = CongestionANN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # weights per class
        # free flowing: 1/314 = 0.003
        # heavy delay:  1/47  = 0.021
        class_counts = np.bincount(y)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # create dataloaders
        train_loader, val_loader = self.ann_dataloaders(X_train, y_train, X_val, y_val)
        
        # training loop
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.epochs):
            # training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_targets, val_preds)
            
            # update scheduler
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
        
        # get best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.model.eval()
        val_preds = self.predict(X_val)
        train_preds = self.predict(X_train)
        
        train_preds_encoded = self.label_encoder.transform(train_preds)
        val_preds_encoded = self.label_encoder.transform(val_preds)
        
        train_acc = accuracy_score(y_train, train_preds_encoded)
        val_acc = accuracy_score(y_val, val_preds_encoded)
        
        print(f"\nTraining accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        print("\nValidation Classification Report:")
        print(classification_report(
            y_val, 
            val_preds_encoded,
            target_names=self.label_encoder.classes_
        ))
        
        # Generate confusion matrices and metrics CSVs
        print("\nGenerating confusion matrices and metrics...")
        self._save_confusion_matrix_and_metrics(y_train, train_preds_encoded, f'{model_name}_train')
        self._save_confusion_matrix_and_metrics(y_val, val_preds_encoded, f'{model_name}_val')
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(history['train_loss']),
            'history': history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions (no backprop - just forward pass)
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted class labels
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def save(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'num_classes': self.num_classes,
            'input_dim': self.model.network[0].in_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.hidden_dims = model_data['hidden_dims']
        self.dropout = model_data['dropout']
        self.num_classes = model_data['num_classes']
        
        self.model = CongestionANN(
            input_dim=model_data['input_dim'],
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")


def train_ann_pipeline(train_features_df: pd.DataFrame,
                       target_col: str,
                       model_save_path: str,
                       hidden_dims: List[int] = [256, 128, 64],
                       epochs: int = 200) -> ANNPredictor:
    """
    Complete ANN training pipeline
    
    Args:
        train_features_df: DataFrame with extracted features and labels
        target_col: Name of target column
        model_save_path: Path to save trained model
        hidden_dims: Hidden layer dimensions
        epochs: Maximum training epochs
        
    Returns:
        Trained ANNPredictor
    """
    print(f"\nANN Training pipeline for {target_col}")
    print(f"Dataset shape: {train_features_df.shape}")
    print(f"Target distribution:\n{train_features_df[target_col].value_counts()}")
    
    # Determine model name from save path
    model_name = Path(model_save_path).stem
    
    # initialize predictor
    predictor = ANNPredictor(
        hidden_dims=hidden_dims,
        epochs=epochs
    )
    
    # prep features
    X, y = predictor.prepare_features(
        train_features_df,
        target_col=target_col,
        fit_scaler=True
    )
    
    # train (this will generate confusion matrices and CSVs)
    metrics = predictor.train(X, y, model_name=model_name)
    
    # save
    predictor.save(model_save_path)
    
    return predictor

def train_xgboost_pipeline(train_features_df: pd.DataFrame,
                           target_col: str,
                           model_save_path: str) -> Optional[xgb.XGBClassifier]:
    """
    XGBoost training pipeline - reuses ANNPredictor's prepare_features
    """

    print(f"\nXGBoost Training pipeline for {target_col}")
    print(f"Dataset shape: {train_features_df.shape}")
    print(f"Target distribution:\n{train_features_df[target_col].value_counts()}")
    
    # Determine model name from save path
    model_name = Path(model_save_path).stem
    output_dir = Path(model_save_path).parent
    
    # reuse ANNPredictor
    predictor = ANNPredictor()
    X, y = predictor.prepare_features(train_features_df, target_col=target_col, fit_scaler=True)
    
    # split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training XGBoost on {len(X_train)} samples...")
    print(f"Validation set: {len(X_val)} samples")
    
    # weights per sample
    class_counts = np.bincount(y)
    total = len(y)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # train
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights, 
              eval_set=[(X_val, y_val)], verbose=False)
    
    # evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, target_names=predictor.label_encoder.classes_))
    
    # Generate confusion matrices and metrics CSVs
    print("\nGenerating confusion matrices and metrics...")
    predictor._save_confusion_matrix_and_metrics(y_train, train_pred, f'{model_name}_train', output_dir)
    predictor._save_confusion_matrix_and_metrics(y_val, val_pred, f'{model_name}_val', output_dir)
    
    # save
    with open(model_save_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': predictor.scaler, 
                     'label_encoder': predictor.label_encoder}, f)
    print(f"Model saved to {model_save_path}")
    
    return model