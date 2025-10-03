"""
Animation and easing utilities for NavierFlow GUI
"""
import numpy as np
from typing import Callable


class EasingFunctions:
    """Collection of easing functions for smooth animations"""
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear interpolation"""
        return t
    
    @staticmethod
    def ease_in_quad(t: float) -> float:
        """Quadratic ease-in"""
        return t * t
    
    @staticmethod
    def ease_out_quad(t: float) -> float:
        """Quadratic ease-out"""
        return t * (2 - t)
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Quadratic ease-in-out"""
        if t < 0.5:
            return 2 * t * t
        return -1 + (4 - 2 * t) * t
    
    @staticmethod
    def ease_in_cubic(t: float) -> float:
        """Cubic ease-in"""
        return t * t * t
    
    @staticmethod
    def ease_out_cubic(t: float) -> float:
        """Cubic ease-out"""
        t -= 1
        return t * t * t + 1
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Cubic ease-in-out"""
        if t < 0.5:
            return 4 * t * t * t
        t = 2 * t - 2
        return (t * t * t + 2) / 2
    
    @staticmethod
    def ease_in_sine(t: float) -> float:
        """Sinusoidal ease-in"""
        return 1 - np.cos(t * np.pi / 2)
    
    @staticmethod
    def ease_out_sine(t: float) -> float:
        """Sinusoidal ease-out"""
        return np.sin(t * np.pi / 2)
    
    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        """Sinusoidal ease-in-out"""
        return -(np.cos(np.pi * t) - 1) / 2
    
    @staticmethod
    def ease_in_expo(t: float) -> float:
        """Exponential ease-in"""
        return 0 if t == 0 else np.power(2, 10 * (t - 1))
    
    @staticmethod
    def ease_out_expo(t: float) -> float:
        """Exponential ease-out"""
        return 1 if t == 1 else 1 - np.power(2, -10 * t)
    
    @staticmethod
    def ease_in_out_expo(t: float) -> float:
        """Exponential ease-in-out"""
        if t == 0 or t == 1:
            return t
        if t < 0.5:
            return np.power(2, 20 * t - 10) / 2
        return (2 - np.power(2, -20 * t + 10)) / 2
    
    @staticmethod
    def bounce_out(t: float) -> float:
        """Bounce ease-out"""
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375
    
    @staticmethod
    def elastic_out(t: float) -> float:
        """Elastic ease-out"""
        if t == 0 or t == 1:
            return t
        return np.power(2, -10 * t) * np.sin((t - 0.075) * (2 * np.pi) / 0.3) + 1


class Animator:
    """Animation controller for smooth transitions"""
    
    def __init__(self, duration: float = 1.0, easing: Callable = None):
        """
        Initialize animator
        
        Args:
            duration: Animation duration in seconds
            easing: Easing function to use
        """
        self.duration = duration
        self.easing = easing or EasingFunctions.ease_in_out_cubic
        self.start_time = 0.0
        self.current_time = 0.0
        self.is_animating = False
        self.start_value = 0.0
        self.end_value = 1.0
    
    def start(self, start_value: float, end_value: float, current_time: float):
        """Start animation"""
        self.start_value = start_value
        self.end_value = end_value
        self.start_time = current_time
        self.is_animating = True
    
    def update(self, current_time: float) -> float:
        """Update and get current animated value"""
        if not self.is_animating:
            return self.end_value
        
        elapsed = current_time - self.start_time
        if elapsed >= self.duration:
            self.is_animating = False
            return self.end_value
        
        t = elapsed / self.duration
        eased_t = self.easing(t)
        return self.start_value + (self.end_value - self.start_value) * eased_t
    
    def is_active(self) -> bool:
        """Check if animation is active"""
        return self.is_animating


class TransitionManager:
    """Manages multiple simultaneous animations"""
    
    def __init__(self):
        self.animations = {}
    
    def add_animation(self, name: str, animator: Animator):
        """Add an animation"""
        self.animations[name] = animator
    
    def start(self, name: str, start_value: float, end_value: float, current_time: float):
        """Start a named animation"""
        if name in self.animations:
            self.animations[name].start(start_value, end_value, current_time)
    
    def update(self, current_time: float) -> dict:
        """Update all animations and return current values"""
        return {
            name: animator.update(current_time)
            for name, animator in self.animations.items()
        }
    
    def is_any_active(self) -> bool:
        """Check if any animation is active"""
        return any(animator.is_active() for animator in self.animations.values())


class ParticleSystem:
    """Simple particle system for visual effects"""
    
    def __init__(self, max_particles: int = 1000):
        self.max_particles = max_particles
        self.positions = np.zeros((max_particles, 3))
        self.velocities = np.zeros((max_particles, 3))
        self.colors = np.ones((max_particles, 4))
        self.lifetimes = np.zeros(max_particles)
        self.active_particles = 0
    
    def emit(self, position: np.ndarray, velocity: np.ndarray, color: np.ndarray, lifetime: float):
        """Emit a new particle"""
        if self.active_particles < self.max_particles:
            idx = self.active_particles
            self.positions[idx] = position
            self.velocities[idx] = velocity
            self.colors[idx] = color
            self.lifetimes[idx] = lifetime
            self.active_particles += 1
    
    def update(self, dt: float):
        """Update all particles"""
        if self.active_particles == 0:
            return
        
        # Update positions
        self.positions[:self.active_particles] += self.velocities[:self.active_particles] * dt
        
        # Update lifetimes
        self.lifetimes[:self.active_particles] -= dt
        
        # Remove dead particles
        alive = self.lifetimes[:self.active_particles] > 0
        if not np.all(alive):
            alive_indices = np.where(alive)[0]
            self.positions[:len(alive_indices)] = self.positions[alive_indices]
            self.velocities[:len(alive_indices)] = self.velocities[alive_indices]
            self.colors[:len(alive_indices)] = self.colors[alive_indices]
            self.lifetimes[:len(alive_indices)] = self.lifetimes[alive_indices]
            self.active_particles = len(alive_indices)
    
    def get_active_particles(self):
        """Get active particle data"""
        return {
            'positions': self.positions[:self.active_particles].copy(),
            'colors': self.colors[:self.active_particles].copy()
        }


class ColorInterpolator:
    """Smooth color transitions"""
    
    @staticmethod
    def interpolate(color1: tuple, color2: tuple, t: float) -> tuple:
        """Interpolate between two colors"""
        return tuple(
            c1 + (c2 - c1) * t
            for c1, c2 in zip(color1, color2)
        )
    
    @staticmethod
    def interpolate_colormap(value: float, colormap: str = 'viridis') -> tuple:
        """Get color from colormap"""
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        return cmap(value)
