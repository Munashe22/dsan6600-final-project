"""
Utility Functions for Traffic Congestion Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json


def validate_video_file(video_path: str) -> bool:
    """
    Validate that a video file exists and can be opened
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not Path(video_path).exists():
        return False
    
    cap = cv2.VideoCapture(video_path)
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid


def get_video_info(video_path: str) -> Dict:
    """
    Extract metadata from video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return info


def verify_dataset_structure(metadata_path: str, video_base_path: str) -> Dict:
    """
    Verify that all referenced videos exist and are valid
    
    Args:
        metadata_path: Path to metadata CSV
        video_base_path: Base directory for videos
        
    Returns:
        Dictionary with verification results
    """
    metadata = pd.read_csv(metadata_path)
    
    results = {
        'total_segments': len(metadata),
        'valid_segments': 0,
        'missing_videos': [],
        'invalid_videos': []
    }
    
    for idx, row in metadata.iterrows():
        segment_valid = True
        
        for i in range(4):
            video_path = Path(video_base_path) / row[f'cam{i+1}_filename']
            
            if not video_path.exists():
                results['missing_videos'].append(str(video_path))
                segment_valid = False
            elif not validate_video_file(str(video_path)):
                results['invalid_videos'].append(str(video_path))
                segment_valid = False
        
        if segment_valid:
            results['valid_segments'] += 1
    
    return results


def plot_feature_importance(importance_csv: str, top_n: int = 20, output_path: Optional[str] = None):
    """
    Plot feature importance
    
    Args:
        importance_csv: Path to feature importance CSV
        top_n: Number of top features to plot
        output_path: Optional path to save plot
    """
    df = pd.read_csv(importance_csv)
    
    # Get top N features
    top_features = df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['Importance Score'])
    plt.yticks(range(len(top_features)), top_features['Feature Name'])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_predictions(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        class_names: List[str]) -> Dict:
    """
    Evaluate prediction performance
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'accuracy': report['accuracy']
    }


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str],
                         output_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Optional path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_feature_documentation(enter_importance_path: str,
                                 exit_importance_path: str,
                                 output_path: str,
                                 top_n: int = 20):
    """
    Create documentation table for top contributing features
    
    Args:
        enter_importance_path: Path to entrance feature importance CSV
        exit_importance_path: Path to exit feature importance CSV
        output_path: Path to save documentation
        top_n: Number of top features to document
    """
    enter_df = pd.read_csv(enter_importance_path).head(top_n)
    exit_df = pd.read_csv(exit_importance_path).head(top_n)
    
    # Create combined documentation
    doc = {
        'entrance_congestion_top_features': enter_df.to_dict('records'),
        'exit_congestion_top_features': exit_df.to_dict('records'),
        'summary': {
            'total_features': len(enter_df),
            'model_type': 'Random Forest',
            'feature_extraction_method': 'Computer Vision (Background Subtraction + Optical Flow)',
            'key_insights': [
                'Occupancy ratio is the strongest predictor of congestion',
                'Vehicle count provides direct volume measurement',
                'Motion patterns distinguish flowing vs stopped traffic',
                'Temporal features capture congestion buildup over time',
                'Queue length estimation critical for delay classification'
            ]
        }
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(doc, f, indent=2)
    
    print(f"Feature documentation saved to {output_path}")
    
    # Also create markdown version
    md_path = Path(output_path).with_suffix('.md')
    
    with open(md_path, 'w') as f:
        f.write("# Traffic Congestion Prediction - Feature Importance Report\n\n")
        
        f.write("## Entrance Congestion - Top Features\n\n")
        f.write("| Rank | Feature Name | Importance | Contribution | Notes |\n")
        f.write("|------|--------------|------------|--------------|-------|\n")
        for i, row in enter_df.iterrows():
            f.write(f"| {i+1} | {row['Feature Name']} | {row['Importance Score']:.4f} | "
                   f"{row['Normalized Importance']*100:.2f}% | {row['Notes']} |\n")
        
        f.write("\n## Exit Congestion - Top Features\n\n")
        f.write("| Rank | Feature Name | Importance | Contribution | Notes |\n")
        f.write("|------|--------------|------------|--------------|-------|\n")
        for i, row in exit_df.iterrows():
            f.write(f"| {i+1} | {row['Feature Name']} | {row['Importance Score']:.4f} | "
                   f"{row['Normalized Importance']*100:.2f}% | {row['Notes']} |\n")
        
        f.write("\n## Summary\n\n")
        f.write("**Model Type:** Random Forest Classifier\n\n")
        f.write("**Feature Extraction:** Computer Vision (Background Subtraction + Optical Flow)\n\n")
        f.write("**Key Insights:**\n")
        for insight in doc['summary']['key_insights']:
            f.write(f"- {insight}\n")
    
    print(f"Markdown documentation saved to {md_path}")


def load_and_prepare_test_data(test_metadata_path: str) -> pd.DataFrame:
    """
    Load test metadata and prepare for inference
    
    Args:
        test_metadata_path: Path to test metadata CSV
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(test_metadata_path)
    
    # Ensure proper sorting
    df = df.sort_values(['test_period_id', 'minute']).reset_index(drop=True)
    
    return df


def validate_predictions(predictions_df: pd.DataFrame) -> bool:
    """
    Validate prediction format
    
    Args:
        predictions_df: DataFrame with predictions
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = ['test_period_id', 'minute', 
                    'congestion_enter_rating', 'congestion_exit_rating']
    
    if not all(col in predictions_df.columns for col in required_cols):
        print("Missing required columns")
        return False
    
    valid_classes = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    
    if not predictions_df['congestion_enter_rating'].isin(valid_classes).all():
        print("Invalid entrance rating values")
        return False
    
    if not predictions_df['congestion_exit_rating'].isin(valid_classes).all():
        print("Invalid exit rating values")
        return False
    
    return True


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        print(f"{description}: 0/{total} (0.00%)")
    
    def update(self, n: int = 1):
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.2f}%)", end='')
        
        if self.current >= self.total:
            print()  # New line when complete


if __name__ == "__main__":
    print("Utility Functions Module")
    print("Available functions:")
    print("- validate_video_file: Check if video is valid")
    print("- get_video_info: Extract video metadata")
    print("- verify_dataset_structure: Validate entire dataset")
    print("- plot_feature_importance: Visualize feature importance")
    print("- evaluate_predictions: Calculate metrics")
    print("- create_feature_documentation: Generate documentation")