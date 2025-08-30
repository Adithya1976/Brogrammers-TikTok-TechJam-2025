import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import s2sphere
import pandas as pd
import os
import geoclip 
import io
import cv2
import base64
from typing import Optional, Tuple

# ==========================================================================================
# Self-Contained S2 Utility Class
# ==========================================================================================
class S2Utils:
    """
    Correctly maps all 100,000 locations from the training data, where the
    class_id is the row index.
    """
    def init(self, level=6, expected_num_classes=None):
        self.level = level
        self.class_to_latlon_map = self._get_class_to_latlon_map(expected_num_classes)
        self.s2_to_class_map = self._get_s2_to_class_map(self.class_to_latlon_map, level)
        print("Pre-calculating known cell coordinates for distance checks...")
        self.known_coords = np.array([self.class_to_latlon_map[i] for i in range(len(self.class_to_latlon_map))])
        print("✓ Pre-calculation complete.")

    def _get_class_to_latlon_map(self, expected_num_classes):
        package_path = os.path.dirname(geoclip.file)
        source_csv_path = os.path.join(package_path, 'model', 'gps_gallery', 'coordinates_100K.csv')
        df = pd.read_csv(source_csv_path)
        if expected_num_classes is not None and len(df) > expected_num_classes:
            df = df.head(expected_num_classes)
        return {idx: (row['LAT'], row['LON']) for idx, row in df.iterrows()}

    def _get_s2_to_class_map(self, class_to_latlon, level):
        s2_map = {}
        for class_id, (lat, lon) in class_to_latlon.items():
            cell_id = self.latlon_to_cell_id(lat, lon, level)
            if cell_id not in s2_map:
                s2_map[cell_id] = class_id
        return s2_map

    def latlon_to_cell_id(self, lat, lon, level=None):
        if level is None: level = self.level
        return s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(lat, lon)).parent(level).id()

    def cell_id_to_latlon(self, cell_id):
        lat_lng = s2sphere.CellId(int(cell_id)).to_lat_lng()
        return lat_lng.lat().degrees, lat_lng.lng().degrees

    def class_to_cell_id(self, class_id):
        lat, lon = self.class_to_latlon_map[class_id]
        return self.latlon_to_cell_id(lat, lon, self.level)

    def cell_id_to_class(self, cell_id):
        class_id = self.s2_to_class_map.get(cell_id)
        if class_id is not None: return class_id
        print(f"Warning: Target cell {cell_id} not in training set. Finding closest known cell...")
        target_lat, target_lon = self.cell_id_to_latlon(cell_id)
        target_coord = np.array([target_lat, target_lon])
        distances_sq = np.sum((self.known_coords - target_coord) ** 2, axis=1)
        closest_class_idx = np.argmin(distances_sq)
        print(f"--> Using proxy target class: {closest_class_idx}")
        return closest_class_idx

# ==========================================================================================
# Adversarial Noise Generator Class
# ==========================================================================================
class AdversarialNoiseGenerator:
    """
    A self-contained class to generate high-resolution adversarial noise for images
    using the GeoCLIP model.
    """
    def init(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing AdversarialNoiseGenerator on device: {self.device}")
        
        geoclip_instance = geoclip.GeoCLIP()
        self.model = geoclip_instance.to(self.device)
        self.model.eval()

        num_model_classes = self.model.gps_gallery.shape[0]
        print(f"GeoCLIP model loaded, trained on {num_model_classes} locations.")
        self.s2_utils = S2Utils(level=6, expected_num_classes=num_model_classes)
        print("Pre-computing the location feature gallery for adversarial attacks...")
        with torch.no_grad():
            raw_gps_coords = self.model.gps_gallery.to(self.device)
            self.location_feature_gallery = self.model.location_encoder(raw_gps_coords)
        print(f"✓ Location feature gallery computed. Shape: {self.location_feature_gallery.shape}")

        self.preprocess_224 = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _generate_pgd_perturbation(self, image_224_tensor: torch.Tensor,
                                   target_coords: Optional[Tuple[float, float]] = None,
                                   epsilon: float = 0.01, iterations: int = 20) -> torch.Tensor:
        """
        Internal method to generate a low-resolution noise tensor using PGD.
        The entire loop runs on the selected device for maximum speed.
        """
        adversarial_tensor = image_224_tensor.clone().detach()
        alpha = epsilon / 8.0
        
        if target_coords:
            cell_id = self.s2_utils.latlon_to_cell_id(target_coords[0], target_coords[1])
            target_label = torch.tensor([self.s2_utils.cell_id_to_class(cell_id)], device=self.device)
        else:
            with torch.no_grad():
                features = self.model.image_encoder(adversarial_tensor)
                logits = self.model.logit_scale * (features @ self.location_feature_gallery.T)
                original_class = torch.argmax(logits, dim=1).item()
            target_label = torch.tensor([original_class], device=self.device)

        for _ in range(iterations):
            adversarial_tensor.requires_grad = True
            image_features = self.model.image_encoder(adversarial_tensor)
            logits = self.model.logit_scale * (image_features @ self.location_feature_gallery.T)
            
            loss = F.cross_entropy(logits, target_label)
            if not target_coords: loss = -loss

            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad_sign = adversarial_tensor.grad.data.sign()
                if target_coords: adversarial_tensor.data -= alpha * grad_sign
                else: adversarial_tensor.data += alpha * grad_sign
                
                perturbation = torch.clamp(adversarial_tensor.data - image_224_tensor.data, -epsilon, epsilon)
                adversarial_tensor.data = image_224_tensor.data + perturbation
        
        return adversarial_tensor.detach() - image_224_tensor

    def apply_pgd_attack(self, image: Image.Image,
                         epsilon: float = 0.01,
                         iterations: int = 20,
                         target_coords: Optional[Tuple[float, float]] = None) -> Image.Image:
        """
        Applies a PGD adversarial attack and returns a high-resolution adversarial image.
        """
        print(f"Applying {'Targeted' if target_coords else 'Untargeted'} PGD attack...")
        
        image_224_tensor = self.preprocess_224(image).unsqueeze(0).to(self.device)
        
        perturbation_224 = self._generate_pgd_perturbation(
            image_224_tensor, target_coords, epsilon, iterations
        )

        high_res_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device)
        upscaled_perturbation = F.interpolate(
            perturbation_224, size=high_res_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        
        adversarial_high_res_tensor = high_res_tensor + upscaled_perturbation
        
        adversarial_high_res_tensor = torch.clamp(adversarial_high_res_tensor, 0, 1)
        adversarial_image = T.ToPILImage()(adversarial_high_res_tensor.squeeze(0).cpu())
        
        print("✓ High-resolution adversarial image generated.")
        return adversarial_image