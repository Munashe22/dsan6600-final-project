"""
Advanced Feature Extraction using YOLO + CNNs + Traditional CV

This module uses:
1. YOLOv8 for precise vehicle detection and tracking
2. Pre-trained CNNs (ResNet/EfficientNet) for scene understanding
3. Optical flow for motion analysis
4. All features feed into traditional ML (no backprop during inference)
"""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class AdvancedVideoFeatures:
    """Container for advanced extracted features"""
    # YOLO-based vehicle detection
    vehicle_count: float
    car_count: float
    truck_count: float
    motorcycle_count: float
    bus_count: float
    
    # Vehicle tracking features
    avg_vehicle_speed: float
    speed_variance: float
    stopped_vehicles: int
    vehicles_entering: int
    vehicles_exiting: int
    
    # Spatial distribution
    vehicles_in_lanes: Dict[str, int]
    occupancy_by_zone: Dict[str, float]
    
    # CNN scene features
    scene_embedding_mean: float
    scene_embedding_std: float
    scene_complexity: float
    
    # Motion features
    avg_motion_magnitude: float
    motion_entropy: float
    directional_flow: Dict[str, float]
    
    # Congestion indicators
    density_score: float
    queue_length: float
    flow_rate: float
    
    def to_dict(self) -> Dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    result[f"{key}_{subkey}"] = subval
            else:
                result[key] = value
        return result


class YOLOVehicleDetector:
    """Vehicle detection and tracking using YOLOv8"""
    
    def __init__(self, model_size: str = 'n', confidence: float = 0.25):
        """
        Initialize YOLO detector
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            confidence: Detection confidence threshold
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        # Load YOLO model (will auto-download on first use)
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.confidence = confidence
        
        # Vehicle class IDs in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Track vehicles across frames
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected vehicles with bounding boxes and classes
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Only keep vehicle classes
            if class_id in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                detections.append({
                    'class': self.vehicle_classes[class_id],
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return detections
    
    def track_vehicles(self, detections: List[Dict], 
                      prev_detections: List[Dict],
                      max_distance: float = 50.0) -> List[Dict]:
        """
        Simple vehicle tracking between frames
        
        Args:
            detections: Current frame detections
            prev_detections: Previous frame detections
            max_distance: Maximum distance for matching
            
        Returns:
            Detections with tracking IDs
        """
        if not prev_detections:
            # First frame - assign new IDs
            for det in detections:
                det['track_id'] = self.next_vehicle_id
                self.next_vehicle_id += 1
            return detections
        
        # Match current detections to previous ones
        matched = set()
        for det in detections:
            best_match = None
            best_distance = max_distance
            
            for i, prev_det in enumerate(prev_detections):
                if i in matched:
                    continue
                
                # Calculate distance between centers
                dx = det['center'][0] - prev_det['center'][0]
                dy = det['center'][1] - prev_det['center'][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                det['track_id'] = prev_detections[best_match]['track_id']
                matched.add(best_match)
            else:
                # New vehicle
                det['track_id'] = self.next_vehicle_id
                self.next_vehicle_id += 1
        
        return detections


class CNNFeatureExtractor:
    """Extract high-level scene features using pre-trained CNN"""
    
    def __init__(self, model_name: str = 'resnet18'):
        """
        Initialize CNN feature extractor
        
        Args:
            model_name: Pre-trained model ('resnet18', 'resnet50', 'efficientnet_b0')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Remove final classification layer to get features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract CNN features from frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Feature vector
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_tensor)
        
        # Flatten and return as numpy
        features = features.squeeze().cpu().numpy()
        return features


class AdvancedVideoProcessor:
    """
    Advanced video processor combining YOLO, CNN, and traditional CV
    """
    
    def __init__(self,
                 use_yolo: bool = True,
                 use_cnn: bool = True,
                 yolo_model: str = 'n',
                 cnn_model: str = 'resnet18'):
        """
        Initialize advanced processor
        
        Args:
            use_yolo: Whether to use YOLO for detection
            use_cnn: Whether to use CNN for scene features
            yolo_model: YOLO model size
            cnn_model: CNN model name
        """
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.use_cnn = use_cnn
        
        # Initialize detectors
        if self.use_yolo:
            print("Initializing YOLO detector...")
            self.yolo_detector = YOLOVehicleDetector(model_size=yolo_model)
        
        if self.use_cnn:
            print("Initializing CNN feature extractor...")
            self.cnn_extractor = CNNFeatureExtractor(model_name=cnn_model)
        
        # Optical flow for motion
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Background subtractor (backup if YOLO fails)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
    
    def process_video(self, 
                     video_path: str,
                     max_frames: Optional[int] = None,
                     sample_rate: int = 5) -> AdvancedVideoFeatures:
        """
        Process video and extract advanced features
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            sample_rate: Process every Nth frame (for CNN to save compute)
            
        Returns:
            AdvancedVideoFeatures object
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Feature accumulators
        yolo_detections_history = []
        cnn_features_history = []
        motion_history = []
        
        prev_gray = None
        prev_detections = []
        frame_count = 0
        
        # Zone definitions (divide frame into grid)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        zones = self._define_zones(frame_width, frame_height)
        
        # Track vehicles entering/exiting
        vehicle_tracks = defaultdict(list)
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_count += 1
            frame_resized = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            # YOLO Detection
            if self.use_yolo and frame_count % 2 == 0:  # Every 2nd frame
                detections = self.yolo_detector.detect_vehicles(frame_resized)
                detections = self.yolo_detector.track_vehicles(detections, prev_detections)
                yolo_detections_history.append(detections)
                prev_detections = detections
                
                # Track vehicle positions
                for det in detections:
                    vehicle_tracks[det['track_id']].append(det['center'])
            
            # CNN Features
            if self.use_cnn and frame_count % sample_rate == 0:
                cnn_features = self.cnn_extractor.extract_features(frame_resized)
                cnn_features_history.append(cnn_features)
            
            # Optical Flow
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, **self.flow_params
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_history.append({
                    'magnitude': np.mean(mag),
                    'angle': ang,
                    'flow': flow
                })
            
            prev_gray = gray.copy()
        
        cap.release()
        
        # Aggregate features
        features = self._aggregate_advanced_features(
            yolo_detections_history=yolo_detections_history,
            cnn_features_history=cnn_features_history,
            motion_history=motion_history,
            vehicle_tracks=vehicle_tracks,
            zones=zones,
            frame_size=(640, 480)
        )
        
        return features
    
    def _define_zones(self, width: int, height: int) -> Dict[str, Tuple]:
        """Define spatial zones for analysis"""
        return {
            'left': (0, 0, width//3, height),
            'center': (width//3, 0, 2*width//3, height),
            'right': (2*width//3, 0, width, height),
            'top': (0, 0, width, height//3),
            'middle': (0, height//3, width, 2*height//3),
            'bottom': (0, 2*height//3, width, height)
        }
    
    def _aggregate_advanced_features(self,
                                    yolo_detections_history: List,
                                    cnn_features_history: List,
                                    motion_history: List,
                                    vehicle_tracks: Dict,
                                    zones: Dict,
                                    frame_size: Tuple) -> AdvancedVideoFeatures:
        """Aggregate all extracted features"""
        
        # YOLO-based features
        if yolo_detections_history:
            all_detections = [d for frame_dets in yolo_detections_history for d in frame_dets]
            
            vehicle_count = len(yolo_detections_history[-1]) if yolo_detections_history[-1] else 0
            
            # Count by type
            type_counts = defaultdict(int)
            for det in all_detections:
                type_counts[det['class']] += 1
            
            car_count = type_counts.get('car', 0) / max(len(yolo_detections_history), 1)
            truck_count = type_counts.get('truck', 0) / max(len(yolo_detections_history), 1)
            motorcycle_count = type_counts.get('motorcycle', 0) / max(len(yolo_detections_history), 1)
            bus_count = type_counts.get('bus', 0) / max(len(yolo_detections_history), 1)
            
            # Calculate speeds
            speeds = []
            for track_id, positions in vehicle_tracks.items():
                if len(positions) > 1:
                    # Calculate average speed (pixels per frame)
                    distances = [np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) 
                               for i in range(1, len(positions))]
                    if distances:
                        speeds.append(np.mean(distances))
            
            avg_vehicle_speed = np.mean(speeds) if speeds else 0
            speed_variance = np.var(speeds) if speeds else 0
            stopped_vehicles = sum(1 for s in speeds if s < 2.0)
            
            # Entry/Exit detection (simple heuristic based on position change)
            vehicles_entering = 0
            vehicles_exiting = 0
            for track_id, positions in vehicle_tracks.items():
                if len(positions) >= 2:
                    start_y = positions[0][1]
                    end_y = positions[-1][1]
                    if end_y < start_y - 50:  # Moving up (entering)
                        vehicles_entering += 1
                    elif end_y > start_y + 50:  # Moving down (exiting)
                        vehicles_exiting += 1
            
            # Spatial distribution
            vehicles_in_lanes = self._count_vehicles_in_zones(
                yolo_detections_history[-1] if yolo_detections_history else [],
                zones
            )
            
            # Occupancy by zone
            occupancy_by_zone = self._calculate_occupancy_by_zone(
                yolo_detections_history[-1] if yolo_detections_history else [],
                zones,
                frame_size
            )
        else:
            vehicle_count = car_count = truck_count = motorcycle_count = bus_count = 0
            avg_vehicle_speed = speed_variance = stopped_vehicles = 0
            vehicles_entering = vehicles_exiting = 0
            vehicles_in_lanes = {k: 0 for k in zones.keys()}
            occupancy_by_zone = {k: 0.0 for k in zones.keys()}
        
        # CNN features
        if cnn_features_history:
            cnn_array = np.array(cnn_features_history)
            scene_embedding_mean = np.mean(cnn_array)
            scene_embedding_std = np.std(cnn_array)
            # Scene complexity as variance in CNN features
            scene_complexity = np.mean(np.var(cnn_array, axis=0))
        else:
            scene_embedding_mean = scene_embedding_std = scene_complexity = 0
        
        # Motion features
        if motion_history:
            avg_motion_magnitude = np.mean([m['magnitude'] for m in motion_history])
            
            # Motion entropy (uniformity of motion)
            all_angles = np.concatenate([m['angle'].flatten() for m in motion_history])
            hist, _ = np.histogram(all_angles, bins=36, range=(0, 2*np.pi))
            hist = hist / hist.sum()
            motion_entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # Directional flow (average flow in each direction)
            flows = np.array([m['flow'] for m in motion_history])
            directional_flow = {
                'left': np.mean(flows[:, :, 0] < 0),
                'right': np.mean(flows[:, :, 0] > 0),
                'up': np.mean(flows[:, :, 1] < 0),
                'down': np.mean(flows[:, :, 1] > 0)
            }
        else:
            avg_motion_magnitude = motion_entropy = 0
            directional_flow = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        # Congestion indicators
        density_score = vehicle_count / (frame_size[0] * frame_size[1] / 10000)  # Normalized
        queue_length = stopped_vehicles / max(vehicle_count, 1)
        flow_rate = (vehicles_entering + vehicles_exiting) / 2.0
        
        return AdvancedVideoFeatures(
            vehicle_count=vehicle_count,
            car_count=car_count,
            truck_count=truck_count,
            motorcycle_count=motorcycle_count,
            bus_count=bus_count,
            avg_vehicle_speed=avg_vehicle_speed,
            speed_variance=speed_variance,
            stopped_vehicles=stopped_vehicles,
            vehicles_entering=vehicles_entering,
            vehicles_exiting=vehicles_exiting,
            vehicles_in_lanes=vehicles_in_lanes,
            occupancy_by_zone=occupancy_by_zone,
            scene_embedding_mean=scene_embedding_mean,
            scene_embedding_std=scene_embedding_std,
            scene_complexity=scene_complexity,
            avg_motion_magnitude=avg_motion_magnitude,
            motion_entropy=motion_entropy,
            directional_flow=directional_flow,
            density_score=density_score,
            queue_length=queue_length,
            flow_rate=flow_rate
        )
    
    def _count_vehicles_in_zones(self, detections: List[Dict], zones: Dict) -> Dict[str, int]:
        """Count vehicles in each zone"""
        counts = {zone: 0 for zone in zones.keys()}
        
        for det in detections:
            cx, cy = det['center']
            for zone_name, (x1, y1, x2, y2) in zones.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    counts[zone_name] += 1
        
        return counts
    
    def _calculate_occupancy_by_zone(self, detections: List[Dict], 
                                    zones: Dict, frame_size: Tuple) -> Dict[str, float]:
        """Calculate occupancy ratio in each zone"""
        occupancy = {zone: 0.0 for zone in zones.keys()}
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            area = (x2 - x1) * (y2 - y1)
            cx, cy = det['center']
            
            for zone_name, (zx1, zy1, zx2, zy2) in zones.items():
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_area = (zx2 - zx1) * (zy2 - zy1)
                    occupancy[zone_name] += area / zone_area
        
        return occupancy


if __name__ == "__main__":
    print("Advanced Feature Extraction Module")
    print("="*60)
    print("\nCapabilities:")
    print("✓ YOLOv8 vehicle detection and tracking")
    print("✓ Pre-trained CNN scene understanding")
    print("✓ Optical flow motion analysis")
    print("✓ Spatial zone analysis")
    print("✓ Vehicle classification by type")
    print("✓ Speed and trajectory tracking")