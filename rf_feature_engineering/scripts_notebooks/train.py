"""
Complete Training Script for Traffic Congestion Prediction

This script demonstrates the full training workflow:
1. Load video data
2. Extract features
3. Augment training data
4. Train models
5. Evaluate and save
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
import json
from datetime import datetime
from typing import Optional

from feature_extraction import MultiCameraProcessor, extract_temporal_features
from model_training import train_pipeline, CongestionPredictor


class DataAugmentor:
    """
    Generate augmented training data to increase dataset size
    
    Techniques:
    - Temporal jittering (slight shifts in feature values)
    - Synthetic minority oversampling for imbalanced classes
    - Feature perturbation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def augment_dataset(self, 
                       df: pd.DataFrame,
                       target_col: str,
                       augmentation_factor: float = 2.0) -> pd.DataFrame:
        """
        Augment training dataset
        
        Args:
            df: Original DataFrame
            target_col: Target column name
            augmentation_factor: How many times to multiply dataset size
            
        Returns:
            Augmented DataFrame
        """
        print(f"Augmenting dataset with factor {augmentation_factor}...")
        print(f"Original size: {len(df)}")
        
        augmented_dfs = [df]  # Start with original data
        
        # Calculate number of augmented samples needed
        n_augmented = int(len(df) * (augmentation_factor - 1))
        
        # Get feature columns (exclude target and metadata)
        exclude_cols = ['segment_id', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for i in range(n_augmented):
            # Sample a random row
            idx = np.random.randint(0, len(df))
            sample = df.iloc[idx].copy()
            
            # Apply perturbations to features
            for col in feature_cols:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    # Add Gaussian noise (5% of standard deviation)
                    noise_level = 0.05 * df[col].std()
                    sample[col] += np.random.normal(0, noise_level)
            
            # Create new segment ID
            sample['segment_id'] = f"{sample['segment_id']}_aug_{i}"
            
            augmented_dfs.append(pd.DataFrame([sample]))
        
        # Combine all augmented data
        augmented_df = pd.concat(augmented_dfs, ignore_index=True)
        
        print(f"Augmented size: {len(augmented_df)}")
        print(f"Class distribution:")
        print(augmented_df[target_col].value_counts())
        
        return augmented_df
    
    def balance_classes(self, 
                       df: pd.DataFrame,
                       target_col: str) -> pd.DataFrame:
        """
        Balance class distribution through oversampling minority classes
        
        Args:
            df: DataFrame with imbalanced classes
            target_col: Target column name
            
        Returns:
            Balanced DataFrame
        """
        print("Balancing classes...")
        
        # Count samples per class
        class_counts = df[target_col].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        
        for class_label in class_counts.index:
            class_df = df[df[target_col] == class_label]
            
            # Calculate how many samples to add
            n_to_add = max_count - len(class_df)
            
            if n_to_add > 0:
                # Oversample this class
                oversampled = class_df.sample(n=n_to_add, replace=True, random_state=self.random_state)
                
                # Add small perturbations to avoid exact duplicates
                feature_cols = [col for col in df.columns 
                              if col not in ['segment_id', target_col] and 
                              df[col].dtype in [np.float64, np.float32]]
                
                for col in feature_cols:
                    noise = np.random.normal(0, 0.02 * df[col].std(), size=len(oversampled))
                    oversampled[col] = oversampled[col] + noise
                
                balanced_dfs.append(class_df)
                balanced_dfs.append(oversampled)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        print("Balanced class distribution:")
        print(balanced_df[target_col].value_counts())
        
        return balanced_df


def load_training_data(metadata_path: str,
                      video_base_path: str,
                      sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and process training data
    
    Args:
        metadata_path: Path to training metadata CSV
        video_base_path: Base directory for video files
        sample_size: Optional - number of samples to process (for testing)
        
    Returns:
        DataFrame with features and labels
    """
    print("Loading training metadata...")
    metadata = pd.read_csv(metadata_path)
    
    if sample_size:
        print(f"Using sample of {sample_size} segments for testing")
        metadata = metadata.sample(n=min(sample_size, len(metadata)), random_state=42)
    
    print(f"Total segments: {len(metadata)}")
    
    # Initialize video processor
    processor = MultiCameraProcessor(num_cameras=4)
    
    all_features = []
    
    for idx, row in metadata.iterrows():
        if idx % 10 == 0:
            print(f"Processing segment {idx+1}/{len(metadata)}")
        
        # Get video paths for all cameras
        video_paths = [
            str(Path(video_base_path) / row[f'cam{i+1}_filename'])
            for i in range(4)
        ]
        
        # Check if all videos exist
        if not all(Path(p).exists() for p in video_paths):
            print(f"Warning: Missing videos for segment {row['segment_id']}")
            continue
        
        try:
            # Extract features
            features = processor.process_cameras(video_paths, row['segment_id'])
            
            # Add labels
            if 'congestion_enter_rating' in row:
                features['congestion_enter_rating'] = row['congestion_enter_rating']
            if 'congestion_exit_rating' in row:
                features['congestion_exit_rating'] = row['congestion_exit_rating']
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing segment {row['segment_id']}: {e}")
            continue
    
    # Combine all features
    features_df = pd.concat(all_features, ignore_index=True)
    
    print(f"\nFeature extraction complete!")
    print(f"Total samples: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")
    
    return features_df


def train_models(features_df: pd.DataFrame,
                output_dir: str,
                augment_data: bool = True,
                balance_classes: bool = True):
    """
    Train both entrance and exit congestion models
    
    Args:
        features_df: DataFrame with features and labels
        output_dir: Directory to save models
        augment_data: Whether to augment training data
        balance_classes: Whether to balance class distribution
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Data augmentation
    if augment_data:
        augmentor = DataAugmentor()
        
        # Augment for entrance ratings
        print("\n" + "="*60)
        print("AUGMENTING DATA FOR ENTRANCE RATINGS")
        print("="*60)
        features_enter = augmentor.augment_dataset(
            features_df, 
            'congestion_enter_rating',
            augmentation_factor=2.0
        )
        
        if balance_classes:
            features_enter = augmentor.balance_classes(
                features_enter,
                'congestion_enter_rating'
            )
        
        # Augment for exit ratings
        print("\n" + "="*60)
        print("AUGMENTING DATA FOR EXIT RATINGS")
        print("="*60)
        features_exit = augmentor.augment_dataset(
            features_df,
            'congestion_exit_rating', 
            augmentation_factor=2.0
        )
        
        if balance_classes:
            features_exit = augmentor.balance_classes(
                features_exit,
                'congestion_exit_rating'
            )
    else:
        features_enter = features_df.copy()
        features_exit = features_df.copy()
    
    # Train entrance model
    print("\n" + "="*60)
    print("TRAINING ENTRANCE CONGESTION MODEL")
    print("="*60)
    enter_model = train_pipeline(
        features_enter,
        target_col='congestion_enter_rating',
        model_save_path=str(output_path / 'entrance_model.pkl'),
        model_type='random_forest'
    )
    
    # Train exit model
    print("\n" + "="*60)
    print("TRAINING EXIT CONGESTION MODEL")
    print("="*60)
    exit_model = train_pipeline(
        features_exit,
        target_col='congestion_exit_rating',
        model_save_path=str(output_path / 'exit_model.pkl'),
        model_type='random_forest'
    )
    
    # Generate feature importance reports
    print("\n" + "="*60)
    print("GENERATING FEATURE IMPORTANCE REPORTS")
    print("="*60)
    
    enter_importance = enter_model.get_feature_importance_report()
    exit_importance = exit_model.get_feature_importance_report()
    
    enter_importance.to_csv(output_path / 'entrance_feature_importance.csv', index=False)
    exit_importance.to_csv(output_path / 'exit_feature_importance.csv', index=False)
    
    print(f"\nTop 10 features for entrance congestion:")
    print(enter_importance.head(10)[['Feature Name', 'Importance Score']].to_string(index=False))
    
    print(f"\nTop 10 features for exit congestion:")
    print(exit_importance.head(10)[['Feature Name', 'Importance Score']].to_string(index=False))
    
    print(f"\nModels and reports saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train traffic congestion prediction models')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to training metadata CSV')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Base directory containing video files')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples to process (for testing)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-balance', action='store_true',
                       help='Disable class balancing')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAFFIC CONGESTION PREDICTION - TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and process training data
    print("\n" + "="*60)
    print("STEP 1: LOADING AND PROCESSING VIDEO DATA")
    print("="*60)
    features_df = load_training_data(
        args.metadata,
        args.video_dir,
        args.sample_size
    )
    
    # Save extracted features
    features_path = Path(args.output_dir) / 'extracted_features.csv'
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"\nExtracted features saved to {features_path}")
    
    # Train models
    print("\n" + "="*60)
    print("STEP 2: TRAINING MODELS")
    print("="*60)
    train_models(
        features_df,
        args.output_dir,
        augment_data=not args.no_augment,
        balance_classes=not args.no_balance
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    # Example usage (can be run directly or via command line)
    
    # For testing without command line args:
    if False:  # Set to True for direct testing
        print("Running in test mode...")
        features_df = pd.DataFrame({
            'segment_id': ['test_1', 'test_2'],
            'cam1_vehicle_count': [10, 15],
            'cam1_occupancy_ratio': [0.3, 0.5],
            'congestion_enter_rating': ['free flowing', 'light delay'],
            'congestion_exit_rating': ['free flowing', 'light delay']
        })
        
        train_models(features_df, 'models', augment_data=False, balance_classes=False)
    else:
        main()