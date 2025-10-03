"""
Loading animations and progress indicators for NavierFlow
"""
import numpy as np
from typing import Optional, Callable
from PyQt6.QtWidgets import QProgressBar, QLabel, QWidget, QVBoxLayout, QDialog
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush


class CircularProgress(QWidget):
    """Circular progress indicator with smooth animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0
        self.target_progress = 0
        self.animation_speed = 0.05
        self.setMinimumSize(100, 100)
        
        # Colors
        self.background_color = QColor(60, 60, 60)
        self.progress_color = QColor(66, 133, 244)
        self.text_color = QColor(255, 255, 255)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)  # ~60 FPS
    
    def _animate(self):
        """Animate progress smoothly"""
        if abs(self.progress - self.target_progress) > 0.01:
            diff = self.target_progress - self.progress
            self.progress += diff * self.animation_speed
            self.update()
    
    def set_progress(self, value: float):
        """Set target progress (0-100)"""
        self.target_progress = max(0, min(100, value))
    
    def paintEvent(self, event):
        """Paint the circular progress"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get dimensions
        width = self.width()
        height = self.height()
        size = min(width, height)
        
        # Calculate circle dimensions
        line_width = size * 0.1
        radius = (size - line_width) / 2
        center_x = width / 2
        center_y = height / 2
        
        # Draw background circle
        painter.setPen(QPen(self.background_color, line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(
            int(center_x - radius), int(center_y - radius),
            int(radius * 2), int(radius * 2)
        )
        
        # Draw progress arc
        painter.setPen(QPen(self.progress_color, line_width))
        start_angle = 90 * 16  # Start at top
        span_angle = -int(self.progress * 3.6 * 16)  # Clockwise
        painter.drawArc(
            int(center_x - radius), int(center_y - radius),
            int(radius * 2), int(radius * 2),
            start_angle, span_angle
        )
        
        # Draw percentage text
        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(int(size * 0.15))
        font.setBold(True)
        painter.setFont(font)
        text = f"{int(self.progress)}%"
        painter.drawText(
            0, 0, width, height,
            Qt.AlignmentFlag.AlignCenter,
            text
        )


class SpinningLoader(QWidget):
    """Spinning circular loader animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.setMinimumSize(50, 50)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)  # ~60 FPS
    
    def _animate(self):
        """Animate rotation"""
        self.angle = (self.angle + 5) % 360
        self.update()
    
    def paintEvent(self, event):
        """Paint the spinning loader"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get dimensions
        width = self.width()
        height = self.height()
        size = min(width, height)
        center_x = width / 2
        center_y = height / 2
        
        # Draw spinning arcs
        line_width = size * 0.1
        radius = (size - line_width) / 2
        
        painter.translate(center_x, center_y)
        painter.rotate(self.angle)
        
        # Draw multiple arcs with varying opacity
        n_arcs = 8
        for i in range(n_arcs):
            alpha = int(255 * (i + 1) / n_arcs)
            color = QColor(66, 133, 244, alpha)
            painter.setPen(QPen(color, line_width))
            
            arc_angle = 360 / n_arcs
            start_angle = int(i * arc_angle * 16)
            span_angle = int(arc_angle * 0.5 * 16)
            
            painter.drawArc(
                int(-radius), int(-radius),
                int(radius * 2), int(radius * 2),
                start_angle, span_angle
            )


class LoadingDialog(QDialog):
    """Modal loading dialog with progress indicator"""
    
    def __init__(self, parent=None, title: str = "Loading...", 
                 message: str = "Please wait..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(300, 150)
        
        # Setup UI
        layout = QVBoxLayout()
        
        # Message label
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        
        # Progress indicator
        self.progress = CircularProgress()
        layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.setLayout(layout)
    
    def set_progress(self, value: float):
        """Update progress"""
        self.progress.set_progress(value)
    
    def set_message(self, message: str):
        """Update message"""
        self.message_label.setText(message)


class ProgressBarWithLabel(QWidget):
    """Progress bar with label and percentage"""
    
    def __init__(self, parent=None, label: str = "Progress"):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        # Label
        self.label = QLabel(label)
        layout.addWidget(self.label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
    def set_progress(self, value: float):
        """Set progress value (0-100)"""
        self.progress_bar.setValue(int(value))
    
    def set_label(self, text: str):
        """Set label text"""
        self.label.setText(text)


class TaskProgressManager:
    """Manage multiple task progress indicators"""
    
    def __init__(self):
        self.tasks = {}
        self.callbacks = []
    
    def add_task(self, task_id: str, total_steps: int = 100):
        """Add a new task to track"""
        self.tasks[task_id] = {
            'current': 0,
            'total': total_steps,
            'completed': False
        }
        self._notify_callbacks()
    
    def update_task(self, task_id: str, current_step: int):
        """Update task progress"""
        if task_id in self.tasks:
            self.tasks[task_id]['current'] = current_step
            if current_step >= self.tasks[task_id]['total']:
                self.tasks[task_id]['completed'] = True
            self._notify_callbacks()
    
    def complete_task(self, task_id: str):
        """Mark task as completed"""
        if task_id in self.tasks:
            self.tasks[task_id]['current'] = self.tasks[task_id]['total']
            self.tasks[task_id]['completed'] = True
            self._notify_callbacks()
    
    def remove_task(self, task_id: str):
        """Remove a task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._notify_callbacks()
    
    def get_task_progress(self, task_id: str) -> float:
        """Get progress percentage for a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return (task['current'] / task['total']) * 100
        return 0.0
    
    def get_overall_progress(self) -> float:
        """Get overall progress across all tasks"""
        if not self.tasks:
            return 100.0
        
        total_progress = sum(
            task['current'] / task['total']
            for task in self.tasks.values()
        )
        return (total_progress / len(self.tasks)) * 100
    
    def is_complete(self) -> bool:
        """Check if all tasks are complete"""
        return all(task['completed'] for task in self.tasks.values())
    
    def register_callback(self, callback: Callable):
        """Register a callback for progress updates"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self):
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            callback(self.tasks)


class PulseAnimation(QWidget):
    """Pulsing animation effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale = 1.0
        self.growing = True
        self.setMinimumSize(30, 30)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)  # ~60 FPS
    
    def _animate(self):
        """Animate pulse"""
        if self.growing:
            self.scale += 0.02
            if self.scale >= 1.2:
                self.growing = False
        else:
            self.scale -= 0.02
            if self.scale <= 0.8:
                self.growing = True
        self.update()
    
    def paintEvent(self, event):
        """Paint the pulse"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get dimensions
        width = self.width()
        height = self.height()
        size = min(width, height)
        center_x = width / 2
        center_y = height / 2
        
        # Calculate scaled radius
        radius = (size / 2) * self.scale
        
        # Draw circle with gradient
        color = QColor(66, 133, 244, int(150 * (2 - self.scale)))
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(center_x - radius), int(center_y - radius),
            int(radius * 2), int(radius * 2)
        )
