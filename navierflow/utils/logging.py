import logging
import sys
from typing import Optional, TextIO
from datetime import datetime
import os
from pathlib import Path

class SimulationLogger:
    def __init__(self,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 stream: Optional[TextIO] = sys.stdout):
        """
        Initialize simulation logger
        
        Args:
            log_file: Path to log file
            level: Logging level
            stream: Output stream for console logging
        """
        # Create logger
        self.logger = logging.getLogger("NavierFlow")
        self.logger.setLevel(level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_formatter = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        
        # Add file handler if log file is specified
        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
        # Add console handler
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize progress tracking
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0
        
    def start_simulation(self, total_steps: int):
        """
        Start simulation logging
        
        Args:
            total_steps: Total number of simulation steps
        """
        self.start_time = datetime.now()
        self.current_step = 0
        self.total_steps = total_steps
        
        self.logger.info("Starting simulation")
        self.logger.info(f"Total steps: {total_steps}")
        
    def update_progress(self, step: int, message: Optional[str] = None):
        """
        Update simulation progress
        
        Args:
            step: Current simulation step
            message: Optional progress message
        """
        self.current_step = step
        
        # Calculate progress
        progress = (step / self.total_steps) * 100
        
        # Calculate elapsed time
        elapsed = datetime.now() - self.start_time
        
        # Calculate estimated time remaining
        if step > 0:
            time_per_step = elapsed / step
            remaining_steps = self.total_steps - step
            remaining_time = time_per_step * remaining_steps
        else:
            remaining_time = None
            
        # Format progress message
        progress_msg = f"Progress: {progress:.1f}% (Step {step}/{self.total_steps})"
        if remaining_time:
            progress_msg += f" - ETA: {remaining_time}"
            
        if message:
            progress_msg += f" - {message}"
            
        self.logger.info(progress_msg)
        
    def log_error(self, error: Exception, context: Optional[str] = None):
        """
        Log error
        
        Args:
            error: Exception to log
            context: Optional error context
        """
        error_msg = str(error)
        if context:
            error_msg = f"{context}: {error_msg}"
            
        self.logger.error(error_msg, exc_info=True)
        
    def log_warning(self, message: str):
        """
        Log warning
        
        Args:
            message: Warning message
        """
        self.logger.warning(message)
        
    def log_info(self, message: str):
        """
        Log info message
        
        Args:
            message: Info message
        """
        self.logger.info(message)
        
    def log_debug(self, message: str):
        """
        Log debug message
        
        Args:
            message: Debug message
        """
        self.logger.debug(message)
        
    def end_simulation(self, success: bool = True):
        """
        End simulation logging
        
        Args:
            success: Whether simulation completed successfully
        """
        # Calculate total runtime
        runtime = datetime.now() - self.start_time
        
        if success:
            self.logger.info("Simulation completed successfully")
        else:
            self.logger.error("Simulation failed")
            
        self.logger.info(f"Total runtime: {runtime}")
        
    def cleanup(self):
        """Cleanup logger"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
            
class ProgressBar:
    def __init__(self,
                 total: int,
                 width: int = 50,
                 stream: TextIO = sys.stdout):
        """
        Initialize progress bar
        
        Args:
            total: Total number of steps
            width: Progress bar width
            stream: Output stream
        """
        self.total = total
        self.width = width
        self.stream = stream
        self.current = 0
        
    def update(self, step: int):
        """
        Update progress bar
        
        Args:
            step: Current step
        """
        self.current = step
        progress = (step / self.total) * 100
        filled = int(self.width * step / self.total)
        bar = "=" * filled + "-" * (self.width - filled)
        
        self.stream.write(f"\r[{bar}] {progress:.1f}%")
        self.stream.flush()
        
    def finish(self):
        """Finish progress bar"""
        self.stream.write("\n")
        self.stream.flush()
        
class Timer:
    def __init__(self, name: str):
        """
        Initialize timer
        
        Args:
            name: Timer name
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start timer"""
        self.start_time = datetime.now()
        
    def stop(self):
        """Stop timer"""
        self.end_time = datetime.now()
        
    def get_elapsed(self) -> float:
        """
        Get elapsed time in seconds
        
        Returns:
            Elapsed time
        """
        if self.start_time is None:
            return 0.0
            
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop() 