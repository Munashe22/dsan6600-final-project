"""
Advanced Training Script with YOLO + CNN Features from GCS

This script:
1. Downloads videos from GCS in batches
2. Extracts features using YOLO + CNN + Traditional CV
3. Trains Random Forest models
4. Cleans up after each batch to save storage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
from typing import Optional, List

from gcs_batch_processor import GCSBatchProcessor
from model_training import train_pipeline, CongestionPredictor
from train import DataAugmentor


def train_from_gcs(bucket_name: str = 'brb-traffic',
                   metadata_path: Optional[str] = None,
                   output_dir: str = 'models',
                   batch_size: int = 10,
                   use_yolo: bool = True,
                   use_cnn: bool = True,
                   augment_data: bool = True,
                   balance_classes: bool = True,
                   ignore_dirs: List[str] = None,
                   max_batches: int = None):
    """
    Complete training pipeline from GCS data
    
    Args:
        bucket_name: GCS bucket name
        metadata_path: Path to training metadata CSV
        output_dir: Directory to save models
        batch_size: Batch size for processing
        use_yolo: Use YOLO for vehicle detection
        use_cnn: Use CNN for scene features
        augment_data: Augment training data
        balance_classes: Balance class distribution
        ignore_dirs: Directories to ignore (default: ['added/'])
        max_batches: Maximum number of batches to process (None = all)
    """
    print("="*80)
    print("ADVANCED TRAINING PIPELINE - YOLO + CNN + TRADITIONAL ML")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if ignore_dirs is None:
        ignore_dirs = ['added/']  # Default: ignore "added/" directory
    
    print(f"\nIgnoring directories: {ignore_dirs}")
    if max_batches:
        print(f"Limiting to {max_batches} batches (batch_size={batch_size}, total videos={max_batches * batch_size})")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize GCS batch processor
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING GCS BATCH PROCESSOR")
    print("="*80)
    
    processor = GCSBatchProcessor(
        bucket_name=bucket_name,
        batch_size=batch_size,
        max_workers=4,
        use_yolo=use_yolo,
        use_cnn=use_cnn
    )
    
    # Extract features from training data
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING FEATURES FROM TRAINING DATA")
    print("="*80)
    
    if metadata_path:
        # Process training data with labels
        features_df = processor.process_training_data(
            metadata_path=metadata_path,
            output_path=str(output_path / 'extracted_features.csv'),
            max_batches=max_batches
        )
    else:
        # Process all videos (no labels - for unsupervised analysis)
        features_df = processor.process_all_videos(
            prefix='',
            output_path=str(output_path / 'extracted_features.csv'),
            save_interval=5,
            ignore_dirs=ignore_dirs
        )
    
    # Cleanup temporary files
    print("\n" + "="*80)
    print("STEP 3: CLEANING UP TEMPORARY FILES")
    print("="*80)
    processor.cleanup_temp_dir()
    
    if len(features_df) == 0:
        print("Error: No features extracted!")
        return
    
    print(f"\n✓ Feature extraction complete!")
    print(f"Total segments: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")
    
    # Check if we have labels
    has_labels = ('congestion_enter_rating' in features_df.columns and 
                  'congestion_exit_rating' in features_df.columns)
    
    if not has_labels:
        print("\nWarning: No labels found in data. Cannot train supervised models.")
        print("Feature extraction complete. Save features and add labels manually.")
        return
    
    # Check if we have actual features (not just labels)
    reserved_columns = ['segment_id', 'congestion_enter_rating', 'congestion_exit_rating', 
                       'time_segment_id', 'date', 'timestamp_start', 'cycle_phase']
    feature_columns = [col for col in features_df.columns if col not in reserved_columns]
    
    print(f"  - Reserved columns (IDs, labels): {len(features_df.columns) - len(feature_columns)}")
    print(f"  - Actual feature columns: {len(feature_columns)}")
    
    if len(feature_columns) == 0:
        print("\n" + "="*80)
        print("❌ ERROR: NO FEATURES EXTRACTED!")
        print("="*80)
        print("\nThe videos were processed but features weren't properly extracted.")
        print("This usually means:")
        print("  1. Video paths in metadata don't match actual filenames")
        print("  2. Videos failed to process (check errors above)")
        print("  3. Feature extraction code has a bug")
        print("\nDebugging:")
        print(f"  - Check: {output_path / 'extracted_features.csv'}")
        print("  - Should have 200+ columns, not just 3")
        print("  - Re-run with the fixed code (filename matching)")
        return
    
    # Data augmentation
    print("\n" + "="*80)
    print("STEP 4: DATA AUGMENTATION AND PREPROCESSING")
    print("="*80)
    
    if augment_data:
        augmentor = DataAugmentor()
        
        # Augment for entrance ratings
        print("\nAugmenting data for entrance ratings...")
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
        print("\nAugmenting data for exit ratings...")
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
    
    # Train models
    print("\n" + "="*80)
    print("STEP 5: TRAINING ENTRANCE CONGESTION MODEL")
    print("="*80)
    
    enter_model = train_pipeline(
        features_enter,
        target_col='congestion_enter_rating',
        model_save_path=str(output_path / 'entrance_model_advanced.pkl'),
        model_type='random_forest'
    )
    
    print("\n" + "="*80)
    print("STEP 6: TRAINING EXIT CONGESTION MODEL")
    print("="*80)
    
    exit_model = train_pipeline(
        features_exit,
        target_col='congestion_exit_rating',
        model_save_path=str(output_path / 'exit_model_advanced.pkl'),
        model_type='random_forest'
    )
    
    # Generate feature importance reports
    print("\n" + "="*80)
    print("STEP 7: GENERATING FEATURE IMPORTANCE REPORTS")
    print("="*80)
    
    enter_importance = enter_model.get_feature_importance_report()
    exit_importance = exit_model.get_feature_importance_report()
    
    enter_importance.to_csv(output_path / 'entrance_feature_importance_advanced.csv', index=False)
    exit_importance.to_csv(output_path / 'exit_feature_importance_advanced.csv', index=False)
    
    print(f"\nTop 15 features for entrance congestion:")
    print(enter_importance.head(15)[['Feature Name', 'Importance Score', 'Normalized Importance']].to_string(index=False))
    
    print(f"\nTop 15 features for exit congestion:")
    print(exit_importance.head(15)[['Feature Name', 'Importance Score', 'Normalized Importance']].to_string(index=False))
    
    # Save configuration
    config = {
        'bucket_name': bucket_name,
        'batch_size': batch_size,
        'use_yolo': use_yolo,
        'use_cnn': use_cnn,
        'augment_data': augment_data,
        'balance_classes': balance_classes,
        'num_segments': len(features_df),
        'num_features': len(features_df.columns),
        'training_date': datetime.now().isoformat()
    }
    
    with open(output_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - entrance_model_advanced.pkl")
    print("  - exit_model_advanced.pkl")
    print("  - entrance_feature_importance_advanced.csv")
    print("  - exit_feature_importance_advanced.csv")
    print("  - extracted_features.csv")
    print("  - training_config.json")


def quick_test_gcs_connection():
    """Quick test to verify GCS connection"""
    print("Testing GCS connection...")
    print("="*60)
    
    try:
        from google.cloud import storage
        
        # Try anonymous client first (for public buckets)
        print("\n1. Trying anonymous access (for public buckets)...")
        try:
            client = storage.Client.create_anonymous_client()
            bucket = client.bucket('brb-traffic')
            
            # Test by listing one blob
            test = list(bucket.list_blobs(max_results=1))
            
            print("   ✓ Anonymous access successful!")
            auth_method = "anonymous"
        except Exception as e:
            print(f"   ✗ Anonymous access failed: {e}")
            print("\n2. Trying authenticated access...")
            try:
                client = storage.Client()
                bucket = client.bucket('brb-traffic')
                
                # Test by listing one blob
                test = list(bucket.list_blobs(max_results=1))
                
                print("   ✓ Authenticated access successful!")
                auth_method = "authenticated"
            except Exception as e2:
                print(f"   ✗ Authenticated access failed: {e2}")
                print("\n" + "="*60)
                print("AUTHENTICATION FAILED")
                print("="*60)
                print("\nTo fix this, try:")
                print("1. gcloud auth application-default login")
                print("2. export GOOGLE_APPLICATION_CREDENTIALS='/path/to/key.json'")
                print("3. Or use local files: train_from_local.py")
                print("\nSee GCS_AUTH_SETUP.md for detailed instructions")
                return False
        
        # If we got here, connection succeeded
        print(f"\n✓ Successfully connected via {auth_method} access")
        print(f"Bucket: brb-traffic")
        
        # Check added/ directory
        print("\n" + "="*60)
        print("Checking bucket structure...")
        print("="*60)
        
        added_blobs = list(bucket.list_blobs(prefix='added/', max_results=5))
        print(f"\nFiles in 'added/' directory (WILL BE IGNORED): {len(added_blobs)}")
        for blob in added_blobs[:3]:
            print(f"  ✗ {blob.name} - IGNORED")
        
        # Check normanniles directories
        print(f"\nFiles in camera directories (WILL BE USED):")
        total_usable = 0
        for camera in ['normanniles1', 'normanniles2', 'normanniles3', 'normanniles4']:
            cam_blobs = list(bucket.list_blobs(prefix=f'{camera}/', max_results=3))
            # Filter out any in added/
            cam_blobs = [b for b in cam_blobs if not b.name.startswith('added/')]
            
            if cam_blobs:
                print(f"\n  {camera}/: found {len(cam_blobs)}+ files")
                for blob in cam_blobs[:2]:
                    print(f"    ✓ {blob.name}")
                total_usable += len(cam_blobs)
            else:
                print(f"\n  {camera}/: ⚠️  No files found (might all be in 'added/')")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        if total_usable > 0:
            print(f"✓ Found {total_usable}+ usable video files")
            print("✓ Ready to train!")
            print("\nRun: python train_advanced.py --metadata train.csv --bucket brb-traffic")
        else:
            print("⚠️  No usable files found in root camera directories")
            print("All videos might be in 'added/' directory")
            print("\nOptions:")
            print("1. Include 'added/': --ignore-dirs \"\"")
            print("2. Update metadata paths to include 'added/' prefix")
            print("3. Use local files: train_from_local.py")
        print("="*60)
        
        return True
        
    except ImportError:
        print("✗ google-cloud-storage not installed")
        print("\nInstall it:")
        print("  pip install google-cloud-storage")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Train traffic congestion models using GCS data with YOLO + CNN'
    )
    parser.add_argument('--bucket', type=str, default='brb-traffic',
                       choices=['brb-traffic', 'brb-traffic-full'],
                       help='GCS bucket name')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to training metadata CSV (if available locally)')
    parser.add_argument('--output-dir', type=str, default='models_advanced',
                       help='Directory to save trained models')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of videos to process in each batch')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process (for testing/limiting)')
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO detection (faster but less accurate)')
    parser.add_argument('--no-cnn', action='store_true',
                       help='Disable CNN features (faster but less rich features)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-balance', action='store_true',
                       help='Disable class balancing')
    parser.add_argument('--test-connection', action='store_true',
                       help='Test GCS connection and exit')
    parser.add_argument('--ignore-dirs', type=str, nargs='+', default=['added/'],
                       help='Directories to ignore in GCS (default: added/)')
    
    args = parser.parse_args()
    
    # Test connection if requested
    if args.test_connection:
        quick_test_gcs_connection()
        return
    
    # Run training
    train_from_gcs(
        bucket_name=args.bucket,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_yolo=not args.no_yolo,
        use_cnn=not args.no_cnn,
        augment_data=not args.no_augment,
        balance_classes=not args.no_balance,
        ignore_dirs=args.ignore_dirs,
        max_batches=args.max_batches
    )


if __name__ == "__main__":
    # For direct testing
    if False:  # Set to True for testing
        print("Running in test mode...")
        
        # Test GCS connection
        quick_test_gcs_connection()
        
        # Test with small batch
        train_from_gcs(
            bucket_name='brb-traffic',
            batch_size=2,
            use_yolo=True,
            use_cnn=True
        )
    else:
        main()