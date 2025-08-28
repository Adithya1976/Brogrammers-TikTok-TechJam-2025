import time
import psutil
from typing import Dict, List
from collections import deque
import threading

# Try to import torch, but don't fail if it's not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PerformanceMonitor:
    """Monitor server performance for mobile app optimization"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.processing_times = deque(maxlen=max_history)
        self.gpu_usage_history = deque(maxlen=max_history)
        self.cpu_usage_history = deque(maxlen=max_history)
        self.memory_usage_history = deque(maxlen=max_history)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Start background monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_system(self):
        """Background system monitoring"""
        while self._monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_history.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage_history.append(memory.percent)
                
                # GPU usage (if available)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        self.gpu_usage_history.append(gpu_memory)
                    except:
                        pass  # Skip GPU monitoring if there's an error
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def record_processing_time(self, processing_time: float):
        """Record a processing time"""
        self.processing_times.append(processing_time)
        self.request_count += 1
    
    def record_error(self):
        """Record an error"""
        self.error_count += 1
    
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_count / max(self.request_count, 1) * 100, 2),
            "requests_per_minute": round(self.request_count / (uptime / 60), 2) if uptime > 60 else 0
        }
        
        # Processing time stats
        if self.processing_times:
            times = list(self.processing_times)
            stats["processing_time"] = {
                "avg": round(sum(times) / len(times), 2),
                "min": round(min(times), 2),
                "max": round(max(times), 2),
                "recent_avg": round(sum(times[-10:]) / min(len(times), 10), 2)
            }
        
        # System usage stats
        if self.cpu_usage_history:
            cpu_usage = list(self.cpu_usage_history)
            stats["cpu_usage"] = {
                "current": cpu_usage[-1],
                "avg": round(sum(cpu_usage) / len(cpu_usage), 1),
                "max": max(cpu_usage)
            }
        
        if self.memory_usage_history:
            memory_usage = list(self.memory_usage_history)
            stats["memory_usage"] = {
                "current": memory_usage[-1],
                "avg": round(sum(memory_usage) / len(memory_usage), 1),
                "max": max(memory_usage)
            }
        
        if self.gpu_usage_history and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_usage = list(self.gpu_usage_history)
                stats["gpu_usage"] = {
                    "current": gpu_usage[-1],
                    "avg": round(sum(gpu_usage) / len(gpu_usage), 1),
                    "max": max(gpu_usage),
                    "memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
                    "memory_reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
                }
            except:
                pass  # Skip GPU stats if there's an error
        
        return stats
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False

# Global performance monitor
performance_monitor = PerformanceMonitor()