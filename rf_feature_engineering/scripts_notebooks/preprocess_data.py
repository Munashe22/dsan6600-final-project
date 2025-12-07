"""
Data Preprocessor for Actual Competition Format

Converts the actual CSV format (one video per row) to the format 
expected by the training pipeline (one row per time segment with all cameras)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def preprocess_competition_data(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Convert competition CSV format to training format
    
    Input format (actual):
        - One row per video (1 camera view)
        - Columns: time_segment_id, videos, congestion_enter_rating, congestion_exit_rating
        - Camera identified by folder: normanniles1/, normanniles2/, etc.
    
    Output format (for training):
        - One row per time segment (4 camera views)
        - Columns: segment_id, cam1_filename, cam2_filename, cam3_filename, cam4_filename,
                   congestion_enter_rating, congestion_exit_rating
    
    Args:
        csv_path: Path to competition CSV
        output_path: Optional path to save converted CSV
        
    Returns:
        Converted DataFrame
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Extract camera number from video path
    df['camera'] = df['videos'].apply(lambda x: int(x.split('/')[0].replace('normanniles', '')))
    
    print(f"\nCameras found: {sorted(df['camera'].unique())}")
    print(f"Time segments: {df['time_segment_id'].nunique()}")
    
    # Pivot to get one row per time segment with all cameras
    pivoted_data = []
    
    for segment_id in sorted(df['time_segment_id'].unique()):
        segment_data = df[df['time_segment_id'] == segment_id]
        
        # Create row with all camera views
        row = {
            'segment_id': f"seg_{segment_id:04d}",
            'time_segment_id': segment_id,
        }
        
        # Add camera filenames
        for camera_num in range(1, 5):
            camera_data = segment_data[segment_data['camera'] == camera_num]
            if len(camera_data) > 0:
                row[f'cam{camera_num}_filename'] = camera_data.iloc[0]['videos']
            else:
                row[f'cam{camera_num}_filename'] = None
                print(f"Warning: Missing camera {camera_num} for segment {segment_id}")
        
        # Add labels (should be same for all cameras in segment)
        if 'congestion_enter_rating' in segment_data.columns:
            row['congestion_enter_rating'] = segment_data.iloc[0]['congestion_enter_rating']
            row['congestion_exit_rating'] = segment_data.iloc[0]['congestion_exit_rating']
        
        # Add metadata
        if 'date' in segment_data.columns:
            row['date'] = segment_data.iloc[0]['date']
        if 'datetimestamp_start' in segment_data.columns:
            row['timestamp_start'] = segment_data.iloc[0]['datetimestamp_start']
        if 'cycle_phase' in segment_data.columns:
            row['cycle_phase'] = segment_data.iloc[0]['cycle_phase']
        
        pivoted_data.append(row)
    
    # Create DataFrame
    result_df = pd.DataFrame(pivoted_data)
    
    print(f"\nConverted shape: {result_df.shape}")
    print(f"Segments with all 4 cameras: {result_df[[f'cam{i}_filename' for i in range(1,5)]].notna().all(axis=1).sum()}")
    
    # Check for missing cameras
    for i in range(1, 5):
        missing = result_df[f'cam{i}_filename'].isna().sum()
        if missing > 0:
            print(f"Warning: {missing} segments missing camera {i}")
    
    # Save if requested
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved converted data to {output_path}")
    
    return result_df


def preprocess_test_data(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Convert test CSV format (similar structure to training)
    
    For test data, we need to group by test period AND minute to get
    the 15-minute input windows with all 4 cameras
    """
    print(f"Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Original shape: {df.shape}")
    
    # Extract camera number
    df['camera'] = df['videos'].apply(lambda x: int(x.split('/')[0].replace('normanniles', '')))
    
    # Create unique segment ID from test_period and minute
    if 'test_period_id' in df.columns and 'minute' in df.columns:
        df['segment_id'] = df['test_period_id'] + '_min_' + df['minute'].astype(str)
    elif 'time_segment_id' in df.columns:
        df['segment_id'] = 'seg_' + df['time_segment_id'].astype(str)
    else:
        df['segment_id'] = df.index.astype(str)
    
    # Pivot
    pivoted_data = []
    
    for segment_id in df['segment_id'].unique():
        segment_data = df[df['segment_id'] == segment_id]
        
        row = {'segment_id': segment_id}
        
        # Add test period and minute if available
        if 'test_period_id' in segment_data.columns:
            row['test_period_id'] = segment_data.iloc[0]['test_period_id']
        if 'minute' in segment_data.columns:
            row['minute'] = segment_data.iloc[0]['minute']
        if 'time_segment_id' in segment_data.columns:
            row['time_segment_id'] = segment_data.iloc[0]['time_segment_id']
        
        # Add camera filenames
        for camera_num in range(1, 5):
            camera_data = segment_data[segment_data['camera'] == camera_num]
            if len(camera_data) > 0:
                row[f'cam{camera_num}_filename'] = camera_data.iloc[0]['videos']
            else:
                row[f'cam{camera_num}_filename'] = None
        
        pivoted_data.append(row)
    
    result_df = pd.DataFrame(pivoted_data)
    
    print(f"\nConverted shape: {result_df.shape}")
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
    
    return result_df


def quick_check_data_format(csv_path: str):
    """Quick check to see if data needs preprocessing"""
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("DATA FORMAT CHECK")
    print("="*80)
    
    print(f"\nFile: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Check if it's already in the right format
    has_cam_columns = all(f'cam{i}_filename' in df.columns for i in range(1, 5))
    
    if has_cam_columns:
        print("\n✓ Data is already in correct format (cam1_filename, cam2_filename, etc.)")
        print("  → Can use directly with training scripts")
        return True
    else:
        print("\n✗ Data needs preprocessing")
        print("  → Has 'videos' column with one video per row")
        print("  → Run: python preprocess_data.py to convert")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess competition data format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--test', action='store_true',
                       help='Process as test data (has test_period_id and minute)')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check format, do not convert')
    
    args = parser.parse_args()
    
    if args.check_only:
        quick_check_data_format(args.input)
    else:
        if args.test:
            preprocess_test_data(args.input, args.output)
        else:
            preprocess_competition_data(args.input, args.output)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Use the converted file: {args.output}")
        print(f"2. Run training:")
        print(f"   python train_advanced.py --metadata {args.output} --bucket brb-traffic")