"""
Real-time Inference Pipeline for Traffic Congestion Prediction

Handles the deployment scenario:
- 15 minutes of input video data
- 2-minute embargo (operational lag)
- 5-minute prediction window (minutes 18-23)

CRITICAL: No future data usage - predictions must be sequential and causal
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta

from feature_extraction import MultiCameraProcessor, extract_temporal_features
from model_training import CongestionPredictor


class RealTimePredictor:
    """
    Real-time traffic congestion predictor
    
    Implements the temporal constraints:
    - Uses only past data (causality)
    - Respects 2-minute embargo
    - Predicts 5-minute horizon
    """
    
    def __init__(self, 
                 enter_model_path: str,
                 exit_model_path: str,
                 num_cameras: int = 4):
        """
        Initialize real-time predictor
        
        Args:
            enter_model_path: Path to entrance congestion model
            exit_model_path: Path to exit congestion model
            num_cameras: Number of camera views
        """
        self.num_cameras = num_cameras
        self.video_processor = MultiCameraProcessor(num_cameras=num_cameras)
        
        # Load models
        self.enter_predictor = CongestionPredictor()
        self.enter_predictor.load(enter_model_path)
        
        self.exit_predictor = CongestionPredictor()
        self.exit_predictor.load(exit_model_path)
        
        print("Real-time predictor initialized")
        print(f"Loaded entrance model: {enter_model_path}")
        print(f"Loaded exit model: {exit_model_path}")
    
    def process_input_window(self, 
                            video_segments: List[Dict[str, str]],
                            segment_ids: List[str]) -> pd.DataFrame:
        """
        Process 15-minute input window
        
        Args:
            video_segments: List of dicts, each with camera paths for one minute
                           Format: [{'cam1': path, 'cam2': path, ...}, ...]
            segment_ids: List of segment identifiers (one per minute)
            
        Returns:
            DataFrame with features for all segments
        """
        if len(video_segments) != 15:
            raise ValueError(f"Expected 15 segments, got {len(video_segments)}")
        
        all_features = []
        
        for i, (segment, seg_id) in enumerate(zip(video_segments, segment_ids)):
            print(f"\nProcessing minute {i+1}/15: {seg_id}")
            
            # Extract camera paths
            camera_paths = [segment[f'cam{j+1}'] for j in range(self.num_cameras)]
            
            # Process all cameras for this minute
            features = self.video_processor.process_cameras(
                video_paths=camera_paths,
                segment_id=seg_id
            )
            
            all_features.append(features)
        
        # Combine all features
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Add temporal features (trends, rolling stats)
        features_df = extract_temporal_features(features_df, window_size=15)
        
        return features_df
    
    def predict_horizon(self, 
                       features_df: pd.DataFrame,
                       prediction_minutes: int = 5) -> pd.DataFrame:
        """
        Predict congestion for the next N minutes after embargo
        
        Args:
            features_df: DataFrame with features from input window (15 minutes)
            prediction_minutes: Number of minutes to predict (default 5)
            
        Returns:
            DataFrame with predictions
        """
        # Use the last available features (minute 15) to predict future
        # This respects causality - we can only use past information
        last_features = features_df.iloc[[-1]].copy()
        
        predictions = []
        
        # For each minute in prediction horizon
        for minute in range(18, 18 + prediction_minutes):  # Minutes 18-22 (5 minutes)
            # Prepare features (exclude labels and segment_id)
            X_enter, _ = self.enter_predictor.prepare_features(
                last_features, 
                target_col=None,
                fit_scaler=False
            )
            
            X_exit, _ = self.exit_predictor.prepare_features(
                last_features,
                target_col=None, 
                fit_scaler=False
            )
            
            # Make predictions
            enter_pred = self.enter_predictor.predict(X_enter)[0]
            exit_pred = self.exit_predictor.predict(X_exit)[0]
            
            # Get probabilities for confidence
            enter_proba = self.enter_predictor.predict_proba(X_enter)[0]
            exit_proba = self.exit_predictor.predict_proba(X_exit)[0]
            
            predictions.append({
                'minute': minute,
                'congestion_enter_rating': enter_pred,
                'congestion_exit_rating': exit_pred,
                'enter_confidence': max(enter_proba),
                'exit_confidence': max(exit_proba)
            })
            
            # Note: In a more sophisticated approach, we could update features
            # based on predictions and temporal trends, but maintaining causality
        
        return pd.DataFrame(predictions)
    
    def predict_from_videos(self,
                           video_segments: List[Dict[str, str]],
                           segment_ids: List[str]) -> pd.DataFrame:
        """
        Complete prediction pipeline from raw videos
        
        Args:
            video_segments: List of video paths for 15-minute window
            segment_ids: Segment identifiers
            
        Returns:
            DataFrame with predictions for minutes 18-22
        """
        print("="*60)
        print("REAL-TIME PREDICTION PIPELINE")
        print("="*60)
        
        # Step 1: Process input window (15 minutes)
        print("\nStep 1: Processing 15-minute input window...")
        features_df = self.process_input_window(video_segments, segment_ids)
        
        # Step 2: Embargo period (2 minutes) - no processing needed
        print("\nStep 2: Embargo period (2 minutes) - waiting...")
        
        # Step 3: Make predictions (5 minutes ahead)
        print("\nStep 3: Predicting minutes 18-22...")
        predictions = self.predict_horizon(features_df, prediction_minutes=5)
        
        print("\n" + "="*60)
        print("PREDICTIONS COMPLETE")
        print("="*60)
        print(predictions.to_string(index=False))
        
        return predictions


class BatchInference:
    """Batch inference for test set evaluation"""
    
    def __init__(self, 
                 enter_model_path: str,
                 exit_model_path: str):
        """
        Initialize batch inference
        
        Args:
            enter_model_path: Path to entrance model
            exit_model_path: Path to exit model
        """
        self.predictor = RealTimePredictor(enter_model_path, exit_model_path)
    
    def predict_test_set(self,
                        test_metadata: pd.DataFrame,
                        video_base_path: str,
                        output_path: str):
        """
        Run predictions on entire test set
        
        Args:
            test_metadata: DataFrame with test set information
            video_base_path: Base directory containing video files
            output_path: Path to save predictions
        """
        all_predictions = []
        
        # Group by test period (each period has 15 input + 5 output minutes)
        test_periods = test_metadata.groupby('test_period_id')
        
        for period_id, period_data in test_periods:
            print(f"\n{'='*60}")
            print(f"Processing test period: {period_id}")
            print(f"{'='*60}")
            
            # Get input window (minutes 1-15)
            input_data = period_data[period_data['minute'] <= 15].sort_values('minute')
            
            if len(input_data) != 15:
                print(f"Warning: Expected 15 input minutes, got {len(input_data)}")
                continue
            
            # Prepare video paths
            video_segments = []
            segment_ids = []
            
            for _, row in input_data.iterrows():
                segment = {
                    f'cam{i+1}': Path(video_base_path) / row[f'cam{i+1}_filename']
                    for i in range(4)
                }
                video_segments.append(segment)
                segment_ids.append(row['segment_id'])
            
            # Make predictions
            try:
                predictions = self.predictor.predict_from_videos(
                    video_segments, segment_ids
                )
                
                predictions['test_period_id'] = period_id
                all_predictions.append(predictions)
                
            except Exception as e:
                print(f"Error processing period {period_id}: {e}")
                continue
        
        # Combine all predictions
        if all_predictions:
            final_predictions = pd.concat(all_predictions, ignore_index=True)
            final_predictions.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")
            return final_predictions
        else:
            print("No predictions generated")
            return pd.DataFrame()


def create_submission(predictions_df: pd.DataFrame, 
                     submission_template_path: str,
                     output_path: str):
    """
    Create submission file in required format
    
    Args:
        predictions_df: DataFrame with predictions
        submission_template_path: Path to submission template
        output_path: Path to save submission
    """
    # Load template
    template = pd.read_csv(submission_template_path)
    
    # Merge predictions with template
    # This depends on the exact submission format required
    # Adjust column names and structure as needed
    
    submission = template.copy()
    
    # Map predictions to submission format
    for idx, row in predictions_df.iterrows():
        # Find matching rows in submission
        mask = (submission['test_period_id'] == row['test_period_id']) & \
               (submission['minute'] == row['minute'])
        
        submission.loc[mask, 'congestion_enter_rating'] = row['congestion_enter_rating']
        submission.loc[mask, 'congestion_exit_rating'] = row['congestion_exit_rating']
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file created: {output_path}")


if __name__ == "__main__":
    print("Real-Time Inference Pipeline")
    print("="*60)
    print("\nDeployment Scenario:")
    print("  Input: 15 minutes of video (minutes 1-15)")
    print("  Embargo: 2 minutes (minutes 16-17)")
    print("  Output: 5 minutes prediction (minutes 18-22)")
    print("\nKey Constraints:")
    print("  ✓ Causal predictions (no future data)")
    print("  ✓ Sequential processing")
    print("  ✓ No backpropagation during inference")
    print("  ✓ Real-time compatible")