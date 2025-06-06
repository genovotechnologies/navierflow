from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import psutil
import numpy as np
from datetime import datetime

class MetricType(Enum):
    """Performance metric types"""
    TIME = "time"
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    IO = "io"
    NETWORK = "network"

@dataclass
class PerformanceMetric:
    """Performance metric"""
    type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation"""
        metric_str = f"{self.type.value.upper()} - {self.name}: {self.value} {self.unit}"
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            metric_str += f" ({context_str})"
            
        return metric_str

class PerformanceMonitor:
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: List[PerformanceMetric] = []
        self.start_time = None
        self.process = psutil.Process()
        
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        
    def measure_time(self,
                    name: str,
                    context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure time
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        # Get current time
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.TIME,
            name=name,
            value=elapsed,
            unit="seconds",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def measure_memory(self,
                      name: str,
                      context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure memory usage
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        # Get memory info
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.MEMORY,
            name=name,
            value=memory_mb,
            unit="MB",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def measure_cpu(self,
                   name: str,
                   context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure CPU usage
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        # Get CPU usage
        cpu_percent = self.process.cpu_percent()
        
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.CPU,
            name=name,
            value=cpu_percent,
            unit="%",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def measure_gpu(self,
                   name: str,
                   context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure GPU usage
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_percent = info.used / info.total * 100
        except:
            gpu_percent = 0.0
            
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.GPU,
            name=name,
            value=gpu_percent,
            unit="%",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def measure_io(self,
                  name: str,
                  context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure I/O operations
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        # Get I/O counters
        io_counters = self.process.io_counters()
        io_bytes = io_counters.read_bytes + io_counters.write_bytes
        
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.IO,
            name=name,
            value=io_bytes,
            unit="bytes",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def measure_network(self,
                       name: str,
                       context: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """
        Measure network usage
        
        Args:
            name: Metric name
            context: Optional context
            
        Returns:
            Performance metric
        """
        # Get network counters
        net_io = psutil.net_io_counters()
        net_bytes = net_io.bytes_sent + net_io.bytes_recv
        
        # Create metric
        metric = PerformanceMetric(
            type=MetricType.NETWORK,
            name=name,
            value=net_bytes,
            unit="bytes",
            timestamp=datetime.now(),
            context=context
        )
        
        # Add metric
        self.metrics.append(metric)
        
        return metric
        
    def get_metrics(self,
                   type: Optional[MetricType] = None,
                   name: Optional[str] = None) -> List[PerformanceMetric]:
        """
        Get metrics
        
        Args:
            type: Optional metric type
            name: Optional metric name
            
        Returns:
            List of metrics
        """
        # Filter metrics
        filtered_metrics = self.metrics
        
        if type is not None:
            filtered_metrics = [
                metric for metric in filtered_metrics
                if metric.type == type
            ]
            
        if name is not None:
            filtered_metrics = [
                metric for metric in filtered_metrics
                if metric.name == name
            ]
            
        return filtered_metrics
        
    def clear_metrics(self):
        """Clear metrics"""
        self.metrics.clear()
        
    def get_summary(self) -> str:
        """
        Get performance summary
        
        Returns:
            Performance summary
        """
        if not self.metrics:
            return "No performance metrics"
            
        # Group metrics by type
        metrics_by_type = {
            type: [
                metric for metric in self.metrics
                if metric.type == type
            ]
            for type in MetricType
        }
        
        # Create summary
        summary = "Performance Summary:\n"
        for type in MetricType:
            metrics = metrics_by_type[type]
            if metrics:
                summary += f"\n{type.value.upper()}:\n"
                
                # Group metrics by name
                metrics_by_name = {}
                for metric in metrics:
                    if metric.name not in metrics_by_name:
                        metrics_by_name[metric.name] = []
                    metrics_by_name[metric.name].append(metric)
                    
                # Add statistics for each metric
                for name, name_metrics in metrics_by_name.items():
                    values = [metric.value for metric in name_metrics]
                    summary += f"  {name}:\n"
                    summary += f"    Min: {min(values):.2f} {name_metrics[0].unit}\n"
                    summary += f"    Max: {max(values):.2f} {name_metrics[0].unit}\n"
                    summary += f"    Mean: {np.mean(values):.2f} {name_metrics[0].unit}\n"
                    summary += f"    Std: {np.std(values):.2f} {name_metrics[0].unit}\n"
                    
        return summary
        
    def save_metrics(self, filename: str):
        """
        Save metrics to file
        
        Args:
            filename: Output filename
        """
        import json
        
        # Convert metrics to dictionary
        metrics_dict = []
        for metric in self.metrics:
            metric_dict = {
                "type": metric.type.value,
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "context": metric.context
            }
            metrics_dict.append(metric_dict)
            
        # Save metrics
        with open(filename, "w") as f:
            json.dump(metrics_dict, f, indent=4)
            
    def load_metrics(self, filename: str):
        """
        Load metrics from file
        
        Args:
            filename: Input filename
        """
        import json
        
        # Load metrics
        with open(filename, "r") as f:
            metrics_dict = json.load(f)
            
        # Convert metrics from dictionary
        self.metrics = []
        for metric_dict in metrics_dict:
            metric = PerformanceMetric(
                type=MetricType(metric_dict["type"]),
                name=metric_dict["name"],
                value=metric_dict["value"],
                unit=metric_dict["unit"],
                timestamp=datetime.fromisoformat(metric_dict["timestamp"]),
                context=metric_dict["context"]
            )
            self.metrics.append(metric) 