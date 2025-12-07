"""
Google Cloud Storage Batch Processor

Downloads videos from GCS in batches, processes them, and deletes local copies
to prevent storage overflow. Designed for memory-efficient processing.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import time
from tqdm import tqdm

from advanced_feature_extraction import AdvancedVideoProcessor


class GCSBatchProcessor:
    """
    Batch processor for GCS video data with automatic cleanup
    
    Features:
    - Downloads videos in configurable batch sizes
    - Processes and extracts features
    - Automatically deletes local files after processing
    - Parallel downloads for efficiency
    - Progress tracking
    """
    
    def __init__(self,
                 bucket_name: str = 'brb-traffic',
                 temp_dir: Optional[str] = None,
                 batch_size: int = 10,
                 max_workers: int = 4,
                 use_yolo: bool = True,
                 use_cnn: bool = True):
        """
        Initialize GCS batch processor
        
        Args:
            bucket_name: GCS bucket name ('brb-traffic' or 'brb-traffic-full')
            temp_dir: Temporary directory for downloads (auto-created if None)
            batch_size: Number of video segments to process at once
            max_workers: Number of parallel download threads
            use_yolo: Use YOLO for detection
            use_cnn: Use CNN for scene features
        """
        self.bucket_name = bucket_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Create temporary directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='traffic_videos_'))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using temporary directory: {self.temp_dir}")
        print(f"Batch size: {batch_size} segments")
        
        # Initialize GCS client
        try:
            # First try anonymous access (for public buckets)
            print("Trying anonymous access to public bucket...")
            self.storage_client = storage.Client.create_anonymous_client()
            self.bucket = self.storage_client.bucket(bucket_name)
            
            # Test if it works by listing one blob
            test_blobs = list(self.bucket.list_blobs(max_results=1))
            
            print(f"✓ Connected to GCS bucket via anonymous access: {bucket_name}")
        except Exception as e:
            print(f"Anonymous access failed: {e}")
            try:
                # Try authenticated access
                print("Trying authenticated access...")
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(bucket_name)
                
                # Test if it works
                test_blobs = list(self.bucket.list_blobs(max_results=1))
                
                print(f"✓ Connected to GCS bucket via authenticated access: {bucket_name}")
            except Exception as e2:
                print(f"Warning: Could not connect to GCS: {e2}")
                print("Running in offline mode - will use local files if available")
                self.storage_client = None
                self.bucket = None
        
        # Initialize video processor
        print("\nInitializing advanced video processor...")
        self.video_processor = AdvancedVideoProcessor(
            use_yolo=use_yolo,
            use_cnn=use_cnn,
            yolo_model='n',  # Fastest YOLO model
            cnn_model='resnet18'  # Efficient CNN
        )
        print("✓ Processor initialized")
    
    def list_videos(self, prefix: str = '', ignore_dirs: List[str] = None) -> List[str]:
        """
        List all videos in GCS bucket
        
        Args:
            prefix: Prefix to filter blobs (e.g., 'normanniles1/')
            ignore_dirs: Directories to ignore (e.g., ['added/', 'test/'])
            
        Returns:
            List of blob paths (excluding ignored directories)
        """
        if self.bucket is None:
            return []
        
        if ignore_dirs is None:
            ignore_dirs = ['added/']  # Default: ignore "added/" directory
        
        blobs = self.bucket.list_blobs(prefix=prefix)
        video_paths = []
        
        for blob in blobs:
            # Skip if not a video file
            if not blob.name.endswith(('.mp4', '.avi', '.mov')):
                continue
            
            # Skip if in ignored directory
            should_skip = False
            for ignore_dir in ignore_dirs:
                if blob.name.startswith(ignore_dir):
                    should_skip = True
                    break
            
            if not should_skip:
                video_paths.append(blob.name)
        
        return video_paths
    
    def download_batch(self, 
                      blob_paths: List[str],
                      show_progress: bool = True) -> List[str]:
        """
        Download a batch of videos from GCS
        
        Args:
            blob_paths: List of blob paths to download
            show_progress: Show progress bar
            
        Returns:
            List of local file paths
        """
        if self.bucket is None:
            raise RuntimeError(
                "Not connected to GCS! Cannot download videos.\n"
                "Please check:\n"
                "1. Is google-cloud-storage installed? pip install google-cloud-storage\n"
                "2. Do you have credentials? export GOOGLE_APPLICATION_CREDENTIALS=key.json\n"
                "3. Is the bucket public or do you have access?\n"
                "4. Try: python train_advanced.py --test-connection\n\n"
                "Alternative: Use local files with train_from_local.py instead"
            )
        
        local_paths = []
        
        def download_single(blob_path: str) -> Tuple[str, str]:
            """Download single blob"""
            local_path = self.temp_dir / Path(blob_path).name
            blob = self.bucket.blob(blob_path)
            
            try:
                blob.download_to_filename(str(local_path))
                return blob_path, str(local_path)
            except Exception as e:
                print(f"Error downloading {blob_path}: {e}")
                return blob_path, None
        
        # Parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(download_single, path) for path in blob_paths]
            
            if show_progress:
                iterator = tqdm(as_completed(futures), total=len(blob_paths), 
                              desc="Downloading")
            else:
                iterator = as_completed(futures)
            
            for future in iterator:
                blob_path, local_path = future.result()
                if local_path:
                    local_paths.append(local_path)
        
        return local_paths
    
    def process_batch(self, local_paths: List[str]) -> pd.DataFrame:
        """
        Process a batch of downloaded videos
        
        Args:
            local_paths: List of local video file paths
            
        Returns:
            DataFrame with extracted features
        """
        all_features = []
        
        for video_path in tqdm(local_paths, desc="Processing videos"):
            try:
                # Extract features
                features = self.video_processor.process_video(
                    video_path,
                    max_frames=None,
                    sample_rate=5
                )
                
                feature_dict = features.to_dict()
                feature_dict['video_path'] = video_path
                feature_dict['video_name'] = Path(video_path).name
                
                all_features.append(feature_dict)
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        
        return pd.DataFrame(all_features)
    
    def cleanup_batch(self, local_paths: List[str]):
        """
        Delete downloaded videos to free up space
        
        Args:
            local_paths: List of local file paths to delete
        """
        for path in local_paths:
            try:
                if Path(path).exists():
                    Path(path).unlink()
            except Exception as e:
                print(f"Warning: Could not delete {path}: {e}")
        
        print(f"✓ Cleaned up {len(local_paths)} files")
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get current storage usage of temp directory"""
        total_size = sum(f.stat().st_size for f in self.temp_dir.rglob('*') if f.is_file())
        
        return {
            'total_mb': total_size / (1024 * 1024),
            'total_gb': total_size / (1024 * 1024 * 1024),
            'num_files': len(list(self.temp_dir.rglob('*')))
        }
    
    def process_all_videos(self,
                          video_list: Optional[List[str]] = None,
                          prefix: str = '',
                          output_path: str = 'extracted_features.csv',
                          save_interval: int = 5,
                          ignore_dirs: List[str] = None) -> pd.DataFrame:
        """
        Process all videos in batches with automatic cleanup
        
        Args:
            video_list: List of video blob paths (if None, lists all from bucket)
            prefix: GCS prefix filter
            output_path: Path to save features incrementally
            save_interval: Save features every N batches
            ignore_dirs: Directories to ignore (default: ['added/'])
            
        Returns:
            DataFrame with all extracted features
        """
        print("="*60)
        print("BATCH PROCESSING PIPELINE")
        print("="*60)
        
        if ignore_dirs is None:
            ignore_dirs = ['added/']  # Default: ignore "added/" directory
        
        # Get video list
        if video_list is None:
            print(f"\nListing videos from GCS (prefix: '{prefix}')...")
            print(f"Ignoring directories: {ignore_dirs}")
            video_list = self.list_videos(prefix=prefix, ignore_dirs=ignore_dirs)
        
        print(f"Total videos to process: {len(video_list)}")
        
        if len(video_list) == 0:
            print("\nWARNING: No videos found!")
            print("This might mean:")
            print("  1. All videos are in ignored directories (e.g., 'added/')")
            print("  2. The prefix doesn't match any files")
            print("  3. The bucket is empty")
            print("\nTo see what's actually in the bucket:")
            print("  python train_advanced.py --test-connection")
            return pd.DataFrame()
        
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {len(video_list) // self.batch_size + 1}")
        
        # Process in batches
        all_features = []
        batch_count = 0
        
        for i in range(0, len(video_list), self.batch_size):
            batch_count += 1
            batch = video_list[i:i + self.batch_size]
            
            print(f"\n{'='*60}")
            print(f"BATCH {batch_count}/{len(video_list) // self.batch_size + 1}")
            print(f"{'='*60}")
            print(f"Videos in batch: {len(batch)}")
            
            # Check storage before download
            storage_before = self.get_storage_usage()
            print(f"Storage usage: {storage_before['total_mb']:.2f} MB "
                  f"({storage_before['num_files']} files)")
            
            # Download batch
            print("\n1. Downloading batch...")
            local_paths = self.download_batch(batch, show_progress=True)
            
            if not local_paths:
                print("No files downloaded, skipping batch")
                continue
            
            # Check storage after download
            storage_after = self.get_storage_usage()
            print(f"Storage usage: {storage_after['total_mb']:.2f} MB "
                  f"(+{storage_after['total_mb'] - storage_before['total_mb']:.2f} MB)")
            
            # Process batch
            print("\n2. Processing batch...")
            batch_features = self.process_batch(local_paths)
            
            if len(batch_features) > 0:
                all_features.append(batch_features)
                print(f"✓ Extracted features from {len(batch_features)} videos")
            
            # Save incrementally
            if batch_count % save_interval == 0 and all_features:
                print(f"\n3. Saving checkpoint...")
                combined_features = pd.concat(all_features, ignore_index=True)
                combined_features.to_csv(output_path, index=False)
                print(f"✓ Saved {len(combined_features)} feature sets to {output_path}")
            
            # Cleanup batch
            print("\n4. Cleaning up batch...")
            self.cleanup_batch(local_paths)
            
            # Final storage check
            storage_final = self.get_storage_usage()
            print(f"Storage usage after cleanup: {storage_final['total_mb']:.2f} MB")
        
        # Final save
        if all_features:
            print(f"\n{'='*60}")
            print("SAVING FINAL RESULTS")
            print(f"{'='*60}")
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_features.to_csv(output_path, index=False)
            print(f"✓ Saved {len(combined_features)} total feature sets to {output_path}")
            
            return combined_features
        else:
            print("No features extracted")
            return pd.DataFrame()
    
    def process_training_data(self,
                             metadata_path: str,
                             output_path: str = 'training_features.csv',
                             max_batches: int = None) -> pd.DataFrame:
        """
        Process training data from metadata file
        
        Args:
            metadata_path: Path to training metadata CSV
            output_path: Path to save extracted features
            max_batches: Maximum number of batches to process (None = all)
            
        Returns:
            DataFrame with features and labels
        """
        print("="*60)
        print("PROCESSING TRAINING DATA")
        print("="*60)
        
        # Load metadata
        metadata = pd.read_csv(metadata_path)
        print(f"\nLoaded metadata: {len(metadata)} segments")
        print(f"Cameras per segment: 4")
        print(f"Total videos to process: {len(metadata) * 4}")
        
        # Get unique camera filenames
        all_videos = []
        for camera in range(1, 5):
            all_videos.extend(metadata[f'cam{camera}_filename'].tolist())
        
        all_videos = list(set(all_videos))  # Remove duplicates
        print(f"Unique videos: {len(all_videos)}")
        
        # Limit videos if max_batches specified
        if max_batches:
            max_videos = max_batches * self.batch_size
            if len(all_videos) > max_videos:
                print(f"\n⚠️  Limiting to {max_batches} batches ({max_videos} videos)")
                all_videos = all_videos[:max_videos]
        
        # Process videos in batches
        video_features = {}
        
        batches_to_process = min(max_batches, len(all_videos) // self.batch_size + 1) if max_batches else (len(all_videos) // self.batch_size + 1)
        
        for i in range(0, len(all_videos), self.batch_size):
            batch_num = i // self.batch_size + 1
            
            # Stop if reached max_batches
            if max_batches and batch_num > max_batches:
                print(f"\n✓ Reached max_batches limit ({max_batches}), stopping")
                break
            
            batch = all_videos[i:i + self.batch_size]
            
            print(f"\nBatch {batch_num}/{batches_to_process}")
            
            # Download
            local_paths = self.download_batch(batch, show_progress=True)
            
            # Process
            batch_df = self.process_batch(local_paths)
            
            # Store by video name
            for _, row in batch_df.iterrows():
                video_features[row['video_name']] = row.to_dict()
            
            # Cleanup
            self.cleanup_batch(local_paths)
        
        # Combine with metadata
        print("\nCombining features with metadata...")
        print(f"Video features extracted: {len(video_features)}")
        print(f"Metadata segments: {len(metadata)}")
        
        all_segment_features = []
        matched_videos = 0
        
        for _, row in metadata.iterrows():
            segment_features = {'segment_id': row.get('segment_id', '')}
            segment_matched = 0
            
            # Get features from each camera
            for camera in range(1, 5):
                video_path_in_metadata = row[f'cam{camera}_filename']
                
                # Extract just the filename (video_features uses filename only)
                video_filename = Path(video_path_in_metadata).name
                
                if video_filename in video_features:
                    cam_features = video_features[video_filename]
                    # Add camera prefix
                    for key, value in cam_features.items():
                        if key not in ['video_path', 'video_name']:
                            segment_features[f'cam{camera}_{key}'] = value
                    segment_matched += 1
                    matched_videos += 1
            
            # Add labels if present
            if 'congestion_enter_rating' in row:
                segment_features['congestion_enter_rating'] = row['congestion_enter_rating']
            if 'congestion_exit_rating' in row:
                segment_features['congestion_exit_rating'] = row['congestion_exit_rating']
            
            all_segment_features.append(segment_features)
        
        print(f"Successfully matched: {matched_videos} camera videos")
        print(f"Segments with at least 1 camera: {sum(1 for s in all_segment_features if len(s) > 3)}")
        
        # Create DataFrame
        features_df = pd.DataFrame(all_segment_features)
        
        # Save
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved training features to {output_path}")
        print(f"Total segments: {len(features_df)}")
        print(f"Total features per segment: {len(features_df.columns)}")
        print(f"  - segment_id: 1")
        print(f"  - labels: 2")
        print(f"  - extracted features: {len(features_df.columns) - 3}")
        
        if len(features_df.columns) <= 3:
            print("\n⚠️  WARNING: No features extracted!")
            print("Only labels found, no video features.")
            print("This means videos were not matched with metadata.")
            print("\nDebugging info:")
            print(f"  - Video features in dictionary: {len(video_features)}")
            print(f"  - Sample video names in features: {list(video_features.keys())[:3]}")
            print(f"  - Sample paths in metadata: {metadata['cam1_filename'].head(3).tolist()}")
        
        return features_df
    
    def cleanup_temp_dir(self):
        """Delete entire temporary directory"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"✓ Deleted temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not delete temp directory: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            # Don't auto-delete in case user wants to inspect
            print(f"\nTemporary directory: {self.temp_dir}")
            print("Run processor.cleanup_temp_dir() to delete")


def download_from_gcs_public(bucket_name: str,
                             prefix: str = '',
                             destination_dir: str = 'data',
                             max_files: Optional[int] = None) -> List[str]:
    """
    Download files from public GCS bucket (no authentication needed)
    
    Args:
        bucket_name: GCS bucket name
        prefix: Prefix to filter files
        destination_dir: Local destination directory
        max_files: Maximum number of files to download
        
    Returns:
        List of downloaded file paths
    """
    from google.cloud import storage
    
    # Anonymous client for public buckets
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    
    # List blobs
    blobs = list(bucket.list_blobs(prefix=prefix))
    video_blobs = [b for b in blobs if b.name.endswith(('.mp4', '.avi', '.mov'))]
    
    if max_files:
        video_blobs = video_blobs[:max_files]
    
    print(f"Found {len(video_blobs)} videos to download")
    
    # Download
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_paths = []
    
    for blob in tqdm(video_blobs, desc="Downloading"):
        local_path = dest_path / Path(blob.name).name
        
        try:
            blob.download_to_filename(str(local_path))
            downloaded_paths.append(str(local_path))
        except Exception as e:
            print(f"Error downloading {blob.name}: {e}")
    
    return downloaded_paths


if __name__ == "__main__":
    print("Google Cloud Storage Batch Processor")
    print("="*60)
    print("\nFeatures:")
    print("✓ Batch download from GCS")
    print("✓ Automatic cleanup to save storage")
    print("✓ Parallel downloads")
    print("✓ Progress tracking")
    print("✓ Incremental saving")
    print("\nGCS Buckets:")
    print("  - brb-traffic (re-encoded, smaller)")
    print("  - brb-traffic-full (full quality, >500GB)")