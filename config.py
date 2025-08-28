import os
import psutil
from typing import Dict, Any

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class ServerConfig:
    """Configuration for RTX 3060 laptop optimization"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self.gpu_available = torch.cuda.is_available()
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else "CPU Only"
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if self.gpu_available else 0
        else:
            self.gpu_available = False
            self.gpu_name = "CPU Only (PyTorch not installed)"
            self.gpu_memory = 0
        
        self.cpu_count = psutil.cpu_count()
        self.ram_total = psutil.virtual_memory().total
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for mobile app"""
        return {
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory / (1024**3), 1) if self.gpu_memory else 0,
            "cpu_cores": self.cpu_count,
            "ram_gb": round(self.ram_total / (1024**3), 1),
            "platform": os.name,
            "optimized_for": "RTX 3060"
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get optimal processing configuration"""
        return {
            "batch_size": 8 if self.gpu_available else 4,
            "max_image_size": (2048, 2048) if self.gpu_available else (1024, 1024),
            "video_frame_limit": 20 if self.gpu_available else 10,
            "concurrent_requests": 4 if self.gpu_available else 2,
            "use_gpu_acceleration": self.gpu_available
        }

# Global config instance
config = ServerConfig()