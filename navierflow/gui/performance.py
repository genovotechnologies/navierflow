"""
Performance monitoring and optimization for NavierFlow
"""
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of performance metrics"""
    FPS = "fps"
    FRAME_TIME = "frame_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    GPU_MEMORY = "gpu_memory"
    SIMULATION_TIME = "simulation_time"
    RENDER_TIME = "render_time"


@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: MetricType


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor
        
        Args:
            history_size: Number of historical values to keep
        """
        self.history_size = history_size
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=history_size)
            for metric_type in MetricType
        }
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Timing contexts
        self._timing_stack = []
        self._timing_results = {}
        
        # Process info
        self.process = psutil.Process()
        
    def start_frame(self):
        """Mark the start of a frame"""
        self.last_frame_time = time.time()
    
    def end_frame(self):
        """Mark the end of a frame and record metrics"""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        
        # Record frame time
        self.record_metric(MetricType.FRAME_TIME, frame_time * 1000)  # ms
        
        # Calculate FPS
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.record_metric(MetricType.FPS, fps)
        
        self.frame_count += 1
    
    def record_metric(self, metric_type: MetricType, value: float):
        """Record a metric value"""
        timestamp = time.time() - self.start_time
        self.metrics[metric_type].append((timestamp, value))
    
    def get_current_metric(self, metric_type: MetricType) -> Optional[float]:
        """Get the most recent metric value"""
        if self.metrics[metric_type]:
            return self.metrics[metric_type][-1][1]
        return None
    
    def get_average_metric(self, metric_type: MetricType, 
                          window: int = 100) -> Optional[float]:
        """Get average metric value over a window"""
        if not self.metrics[metric_type]:
            return None
        
        recent_values = [v for _, v in list(self.metrics[metric_type])[-window:]]
        return np.mean(recent_values)
    
    def get_metric_history(self, metric_type: MetricType) -> List[tuple]:
        """Get full metric history"""
        return list(self.metrics[metric_type])
    
    def update_system_metrics(self):
        """Update system-level metrics"""
        # CPU usage
        cpu_percent = self.process.cpu_percent()
        self.record_metric(MetricType.CPU_USAGE, cpu_percent)
        
        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        self.record_metric(MetricType.MEMORY_USAGE, memory_mb)
        
        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.record_metric(MetricType.GPU_USAGE, gpu.load * 100)
                self.record_metric(MetricType.GPU_MEMORY, gpu.memoryUsed)
        except:
            pass
    
    def start_timing(self, label: str):
        """Start timing a code section"""
        self._timing_stack.append((label, time.perf_counter()))
    
    def end_timing(self, label: str):
        """End timing a code section"""
        if not self._timing_stack or self._timing_stack[-1][0] != label:
            return
        
        start_label, start_time = self._timing_stack.pop()
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        
        if label not in self._timing_results:
            self._timing_results[label] = deque(maxlen=100)
        self._timing_results[label].append(elapsed)
    
    def get_timing_stats(self, label: str) -> Dict[str, float]:
        """Get statistics for a timed section"""
        if label not in self._timing_results:
            return {}
        
        values = list(self._timing_results[label])
        return {
            'mean': np.mean(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'median': np.median(values)
        }
    
    def get_all_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed sections"""
        return {
            label: self.get_timing_stats(label)
            for label in self._timing_results.keys()
        }
    
    def reset(self):
        """Reset all metrics"""
        for metric_type in MetricType:
            self.metrics[metric_type].clear()
        self.frame_count = 0
        self.start_time = time.time()
        self._timing_results.clear()


class PerformanceOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize optimizer
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.target_fps = 60
        self.min_fps = 30
        self.optimization_history = []
        
        # Adjustable parameters
        self.current_quality = 1.0  # 0.0 to 1.0
        self.adaptive_quality = True
    
    def optimize(self) -> Dict[str, any]:
        """
        Analyze performance and suggest optimizations
        
        Returns:
            Dictionary of optimization suggestions
        """
        suggestions = {}
        
        # Get current FPS
        current_fps = self.monitor.get_average_metric(MetricType.FPS, window=30)
        if current_fps is None:
            return suggestions
        
        # Check if optimization is needed
        if current_fps < self.min_fps:
            suggestions['priority'] = 'critical'
            suggestions['message'] = f"FPS too low ({current_fps:.1f}). Immediate optimization needed."
            suggestions['actions'] = self._get_critical_optimizations()
        elif current_fps < self.target_fps:
            suggestions['priority'] = 'moderate'
            suggestions['message'] = f"FPS below target ({current_fps:.1f}/{self.target_fps})."
            suggestions['actions'] = self._get_moderate_optimizations()
        else:
            suggestions['priority'] = 'none'
            suggestions['message'] = f"Performance is good ({current_fps:.1f} FPS)."
            suggestions['actions'] = []
        
        # Check memory usage
        memory = self.monitor.get_current_metric(MetricType.MEMORY_USAGE)
        if memory and memory > 1000:  # Over 1GB
            suggestions['memory_warning'] = True
            suggestions['memory_actions'] = self._get_memory_optimizations()
        
        return suggestions
    
    def _get_critical_optimizations(self) -> List[Dict]:
        """Get critical performance optimizations"""
        return [
            {
                'name': 'Reduce Grid Resolution',
                'description': 'Reduce simulation grid resolution by 50%',
                'impact': 'high',
                'apply': lambda: self._reduce_resolution(0.5)
            },
            {
                'name': 'Disable Advanced Effects',
                'description': 'Disable shadows, ambient occlusion, and motion blur',
                'impact': 'high',
                'apply': lambda: self._disable_effects()
            },
            {
                'name': 'Reduce Update Rate',
                'description': 'Reduce simulation update frequency',
                'impact': 'medium',
                'apply': lambda: self._reduce_update_rate()
            }
        ]
    
    def _get_moderate_optimizations(self) -> List[Dict]:
        """Get moderate performance optimizations"""
        return [
            {
                'name': 'Reduce Visual Quality',
                'description': 'Reduce antialiasing and texture quality',
                'impact': 'medium',
                'apply': lambda: self._reduce_quality()
            },
            {
                'name': 'Optimize Data Structures',
                'description': 'Use more efficient data layouts',
                'impact': 'low',
                'apply': lambda: self._optimize_data()
            }
        ]
    
    def _get_memory_optimizations(self) -> List[Dict]:
        """Get memory optimization suggestions"""
        return [
            {
                'name': 'Clear History',
                'description': 'Clear old performance history',
                'impact': 'low',
                'apply': lambda: self.monitor.reset()
            },
            {
                'name': 'Reduce Cache Size',
                'description': 'Reduce data cache size',
                'impact': 'medium',
                'apply': lambda: self._reduce_cache()
            }
        ]
    
    def _reduce_resolution(self, factor: float):
        """Reduce grid resolution"""
        self.optimization_history.append({
            'action': 'reduce_resolution',
            'factor': factor,
            'timestamp': time.time()
        })
    
    def _disable_effects(self):
        """Disable advanced visual effects"""
        self.optimization_history.append({
            'action': 'disable_effects',
            'timestamp': time.time()
        })
    
    def _reduce_update_rate(self):
        """Reduce update rate"""
        self.optimization_history.append({
            'action': 'reduce_update_rate',
            'timestamp': time.time()
        })
    
    def _reduce_quality(self):
        """Reduce visual quality"""
        self.current_quality *= 0.8
        self.optimization_history.append({
            'action': 'reduce_quality',
            'quality': self.current_quality,
            'timestamp': time.time()
        })
    
    def _optimize_data(self):
        """Optimize data structures"""
        self.optimization_history.append({
            'action': 'optimize_data',
            'timestamp': time.time()
        })
    
    def _reduce_cache(self):
        """Reduce cache size"""
        self.optimization_history.append({
            'action': 'reduce_cache',
            'timestamp': time.time()
        })
    
    def auto_adjust_quality(self):
        """Automatically adjust quality based on performance"""
        if not self.adaptive_quality:
            return
        
        current_fps = self.monitor.get_average_metric(MetricType.FPS, window=10)
        if current_fps is None:
            return
        
        # Adjust quality based on FPS
        if current_fps < self.min_fps:
            self.current_quality = max(0.1, self.current_quality * 0.9)
        elif current_fps > self.target_fps * 1.2:
            self.current_quality = min(1.0, self.current_quality * 1.05)


class FrameRateLimiter:
    """Limit frame rate to target FPS"""
    
    def __init__(self, target_fps: float = 60.0):
        """
        Initialize frame rate limiter
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_frame = time.perf_counter()
    
    def wait(self):
        """Wait to maintain target frame rate"""
        current_time = time.perf_counter()
        elapsed = current_time - self.last_frame
        
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        
        self.last_frame = time.perf_counter()
    
    def set_target_fps(self, fps: float):
        """Set new target FPS"""
        self.target_fps = fps
        self.frame_time = 1.0 / fps


class Profiler:
    """Simple profiler for code sections"""
    
    def __init__(self):
        self.timings = {}
        self._stack = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def start(self, label: str):
        """Start profiling a section"""
        self._stack.append((label, time.perf_counter()))
    
    def stop(self, label: str):
        """Stop profiling a section"""
        if not self._stack or self._stack[-1][0] != label:
            return
        
        start_label, start_time = self._stack.pop()
        elapsed = time.perf_counter() - start_time
        
        if label not in self.timings:
            self.timings[label] = []
        self.timings[label].append(elapsed)
    
    def get_report(self) -> str:
        """Get profiling report"""
        lines = ["Profiling Report", "=" * 50]
        
        for label, times in sorted(self.timings.items()):
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            
            lines.append(f"{label}:")
            lines.append(f"  Total: {total*1000:.2f} ms")
            lines.append(f"  Count: {count}")
            lines.append(f"  Average: {avg*1000:.2f} ms")
            lines.append("")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset profiling data"""
        self.timings.clear()
        self._stack.clear()
