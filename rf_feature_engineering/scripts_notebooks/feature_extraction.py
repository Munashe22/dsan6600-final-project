"""
Video Processing and Feature Extraction for Traffic Congestion Prediction

This module handles video processing and extracts meaningful features that correlate
with traffic congestion levels.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VideoFeatures:
    """Container for extracted video features"""
    # Vehicle counting features
    vehicle_count: float
    entry_count: float
    exit_count: float
    
    # Motion and flow features
    avg_motion_magnitude: float
    motion_density: float
    flow_rate: float
    
    # Congestion indicators
    occupancy_ratio: float
    queue_length: float
    avg_speed_estimate: float
    
    # Temporal features
    stop_frequency: float
    frame_variance: float
    
    # Spatial features
    active_regions: float
    
    def to_dict(self) -> Dict:
        return self.__dict__


class VideoProcessor:
    """Process traffic video and extract congestion-related features"""
    
    def __init__(self, 
                 resize_width: int = 640,
                 resize_height: int = 480,
                 background_history: int = 500,
                 motion_threshold: float = 2.0):
        """
        Initialize video processor
        
        Args:
            resize_width: Width to resize frames to
            resize_height: Height to resize frames to
            background_history: History length for background subtraction
            motion_threshold: Threshold for motion detection
        """
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.motion_threshold = motion_threshold
        
        # Background subtraction for vehicle detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=background_history,
            varThreshold=16,
            detectShadows=True
        )
        
        # Optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> VideoFeatures:
        """
        Process a video file and extract features
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            VideoFeatures object containing extracted features
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Feature accumulators
        motion_magnitudes = []
        occupancy_ratios = []
        frame_variances = []
        vehicle_counts = []
        active_pixel_counts = []
        
        prev_gray = None
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Queue for temporal smoothing
        motion_queue = deque(maxlen=30)
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_count += 1
            
            # Resize frame for efficiency
            frame_resized = cv2.resize(frame, (self.resize_width, self.resize_height))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame_resized)
            
            # Remove shadows (value 127) and keep only foreground (255)
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Calculate occupancy ratio
            occupancy = np.sum(fg_mask > 0) / (self.resize_width * self.resize_height)
            occupancy_ratios.append(occupancy)
            
            # Count connected components as vehicle proxy
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
            # Filter by size to remove noise (vehicles should be reasonably sized)
            valid_components = [i for i in range(1, num_labels) 
                              if stats[i, cv2.CC_STAT_AREA] > 200 and stats[i, cv2.CC_STAT_AREA] < 10000]
            vehicle_counts.append(len(valid_components))
            
            # Calculate optical flow if we have previous frame
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, **self.flow_params
                )
                
                # Calculate motion magnitude
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Only consider motion in foreground regions
                motion_in_fg = mag * (fg_mask / 255.0)
                avg_motion = np.mean(motion_in_fg[motion_in_fg > 0]) if np.any(motion_in_fg > 0) else 0
                motion_magnitudes.append(avg_motion)
                motion_queue.append(avg_motion)
                
                # Count active pixels (pixels with significant motion)
                active_pixels = np.sum(mag > self.motion_threshold)
                active_pixel_counts.append(active_pixels)
            
            # Frame variance as texture/activity indicator
            frame_var = np.var(gray)
            frame_variances.append(frame_var)
            
            prev_gray = gray.copy()
        
        cap.release()
        
        # Aggregate features
        features = self._aggregate_features(
            motion_magnitudes=motion_magnitudes,
            occupancy_ratios=occupancy_ratios,
            frame_variances=frame_variances,
            vehicle_counts=vehicle_counts,
            active_pixel_counts=active_pixel_counts,
            frame_count=frame_count,
            total_frames=total_frames
        )
        
        return features
    
    def _aggregate_features(self,
                           motion_magnitudes: List[float],
                           occupancy_ratios: List[float],
                           frame_variances: List[float],
                           vehicle_counts: List[int],
                           active_pixel_counts: List[int],
                           frame_count: int,
                           total_frames: int) -> VideoFeatures:
        """Aggregate frame-level features into video-level features"""
        
        # Helper function for safe statistics
        def safe_stat(data, func, default=0.0):
            return func(data) if len(data) > 0 else default
        
        # Motion features
        avg_motion = safe_stat(motion_magnitudes, np.mean)
        motion_std = safe_stat(motion_magnitudes, np.std)
        motion_density = safe_stat(active_pixel_counts, np.mean) / (self.resize_width * self.resize_height)
        
        # Occupancy features (strong congestion indicator)
        avg_occupancy = safe_stat(occupancy_ratios, np.mean)
        max_occupancy = safe_stat(occupancy_ratios, max)
        occupancy_std = safe_stat(occupancy_ratios, np.std)
        
        # Vehicle count features
        avg_vehicle_count = safe_stat(vehicle_counts, np.mean)
        max_vehicle_count = safe_stat(vehicle_counts, max)
        
        # Speed estimation (inverse of occupancy with high motion)
        # High occupancy + low motion = congestion
        # Low occupancy + high motion = free flow
        if avg_occupancy > 0:
            speed_estimate = (avg_motion / (avg_occupancy + 0.01))
        else:
            speed_estimate = avg_motion
        
        # Queue length estimation (periods of high occupancy + low motion)
        congested_frames = sum(1 for occ, mot in zip(occupancy_ratios, motion_magnitudes) 
                              if occ > np.percentile(occupancy_ratios, 75) and mot < np.percentile(motion_magnitudes, 25))
        queue_length = congested_frames / max(frame_count, 1)
        
        # Flow rate estimation (balanced motion and vehicle count)
        flow_rate = avg_vehicle_count * avg_motion
        
        # Stop frequency (high variance in motion indicates stop-go patterns)
        stop_frequency = motion_std / (avg_motion + 0.01)
        
        # Frame variance (texture complexity)
        avg_frame_var = safe_stat(frame_variances, np.mean)
        
        # Entry/exit estimation (simplified - in real scenario would need region-based analysis)
        # Using motion patterns in different frame regions
        entry_count = avg_vehicle_count * 0.5  # Placeholder
        exit_count = avg_vehicle_count * 0.5   # Placeholder
        
        # Active regions (percentage of frame with significant activity)
        active_regions = motion_density
        
        return VideoFeatures(
            vehicle_count=avg_vehicle_count,
            entry_count=entry_count,
            exit_count=exit_count,
            avg_motion_magnitude=avg_motion,
            motion_density=motion_density,
            flow_rate=flow_rate,
            occupancy_ratio=avg_occupancy,
            queue_length=queue_length,
            avg_speed_estimate=speed_estimate,
            stop_frequency=stop_frequency,
            frame_variance=avg_frame_var,
            active_regions=active_regions
        )


class MultiCameraProcessor:
    """Process multiple camera views and combine features"""
    
    def __init__(self, num_cameras: int = 4):
        """
        Initialize multi-camera processor
        
        Args:
            num_cameras: Number of camera views
        """
        self.num_cameras = num_cameras
        self.processors = [VideoProcessor() for _ in range(num_cameras)]
    
    def process_cameras(self, 
                       video_paths: List[str],
                       segment_id: str) -> pd.DataFrame:
        """
        Process all camera views for a time segment
        
        Args:
            video_paths: List of paths to video files (one per camera)
            segment_id: Identifier for this time segment
            
        Returns:
            DataFrame with combined features
        """
        if len(video_paths) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} videos, got {len(video_paths)}")
        
        all_features = {}
        
        # Process each camera
        for i, video_path in enumerate(video_paths):
            print(f"Processing camera {i+1}/{self.num_cameras}: {Path(video_path).name}")
            
            try:
                features = self.processors[i].process_video(video_path)
                feature_dict = features.to_dict()
                
                # Add camera prefix
                for key, value in feature_dict.items():
                    all_features[f"cam{i+1}_{key}"] = value
                    
            except Exception as e:
                print(f"Error processing camera {i+1}: {e}")
                # Add default values for failed processing
                for key in VideoFeatures.__annotations__.keys():
                    all_features[f"cam{i+1}_{key}"] = 0.0
        
        # Add aggregate features across cameras
        self._add_aggregate_features(all_features)
        
        # Add segment identifier
        all_features['segment_id'] = segment_id
        
        return pd.DataFrame([all_features])
    
    def _add_aggregate_features(self, features: Dict):
        """Add aggregate features across all cameras"""
        
        # Aggregate similar features across cameras
        feature_types = [
            'vehicle_count', 'entry_count', 'exit_count',
            'avg_motion_magnitude', 'motion_density', 'flow_rate',
            'occupancy_ratio', 'queue_length', 'avg_speed_estimate',
            'stop_frequency', 'frame_variance', 'active_regions'
        ]
        
        for feat_type in feature_types:
            # Collect values from all cameras
            values = [features.get(f"cam{i+1}_{feat_type}", 0.0) 
                     for i in range(self.num_cameras)]
            
            # Add aggregate statistics
            features[f"total_{feat_type}"] = sum(values)
            features[f"avg_{feat_type}"] = np.mean(values)
            features[f"max_{feat_type}"] = max(values)
            features[f"std_{feat_type}"] = np.std(values)


def extract_temporal_features(df: pd.DataFrame, window_size: int = 15) -> pd.DataFrame:
    """
    Extract temporal features from a sequence of video segments
    
    Args:
        df: DataFrame with features from consecutive time segments
        window_size: Number of previous segments to consider
        
    Returns:
        DataFrame with added temporal features
    """
    df = df.copy()
    
    # Get numerical columns (exclude segment_id and labels)
    feature_cols = [col for col in df.columns 
                   if col not in ['segment_id', 'congestion_enter_rating', 'congestion_exit_rating']]
    
    # Add rolling statistics
    for col in feature_cols:
        # Rolling mean
        df[f"{col}_rolling_mean"] = df[col].rolling(window=min(window_size, len(df)), min_periods=1).mean()
        
        # Rolling std
        df[f"{col}_rolling_std"] = df[col].rolling(window=min(window_size, len(df)), min_periods=1).std().fillna(0)
        
        # Trend (difference from rolling mean)
        df[f"{col}_trend"] = df[col] - df[f"{col}_rolling_mean"]
        
        # Change rate
        df[f"{col}_change"] = df[col].diff().fillna(0)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Video Feature Extraction Module")
    print("This module extracts traffic congestion features from video data")
    print("\nKey Features Extracted:")
    print("- Vehicle counts and flow rates")
    print("- Motion and speed patterns")
    print("- Occupancy and queue lengths")
    print("- Stop-go frequency")
    print("- Temporal trends and patterns")