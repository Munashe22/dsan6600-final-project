"""
ANN Model Training Script for Traffic Congestion Prediction
Trains feedforward neural network from extracted video features.
"""
import pandas as pd
from pathlib import Path
from ann_model import train_ann_pipeline, train_xgboost_pipeline

def main():
    # load features
    features_df = pd.read_csv('cnn_model_2/models_advanced/extracted_features.csv')
    print(f"\nLoaded {len(features_df)} samples with {len(features_df.columns)} columns")

    #filter rows
    feature_cols = [c for c in features_df.columns if c not in ['segment_id', 'congestion_enter_rating', 'congestion_exit_rating']]

    # keep rows with 80% of features
    threshold = len(feature_cols) * 0.80
    nan_counts = features_df[feature_cols].isna().sum(axis=1)
    features_df = features_df[nan_counts < threshold]
    print(f"Rows after filtering: {len(features_df)}")

    output_path = Path('models')
    output_path.mkdir(parents=True, exist_ok=True)

    features_enter = features_df.copy()
    features_exit = features_df.copy()

    # ANN 
    print("\n" + "="*50)
    print("TRAINING ANN MODELS")
    print("="*50)
    
    enter_model_ann = train_ann_pipeline(
        features_enter,
        target_col='congestion_enter_rating',
        model_save_path=str(output_path / 'entrance_model_ann.pkl')
    )
    
    exit_model_ann = train_ann_pipeline(
        features_exit,
        target_col='congestion_exit_rating',
        model_save_path=str(output_path / 'exit_model_ann.pkl')
    )
    
    # XGBOOST
    print("\n" + "="*50)
    print("TRAINING XGBOOST MODELS")
    print("="*50)
    
    enter_model_xgb = train_xgboost_pipeline(
        features_enter,
        target_col='congestion_enter_rating',
        model_save_path=str(output_path / 'entrance_model_xgb.pkl')
    )
    
    exit_model_xgb = train_xgboost_pipeline(
        features_exit,
        target_col='congestion_exit_rating',
        model_save_path=str(output_path / 'exit_model_xgb.pkl')
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print("\nGenerated files:")
    print("  - models/entrance_model_ann.pkl")
    print("  - models/exit_model_ann.pkl")
    print("  - models/entrance_model_xgb.pkl")
    print("  - models/exit_model_xgb.pkl")


if __name__ == "__main__":
    main()
