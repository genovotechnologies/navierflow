import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import io
import base64
import cv2

class ExportManager:
    """Manager for exporting simulation results and visualizations"""
    
    def __init__(self):
        self.initialize_state()
        self.supported_formats = {
            'video': ['MP4', 'GIF', 'AVI'],
            'image': ['PNG', 'JPG', 'SVG'],
            'data': ['NPZ', 'JSON', 'CSV']
        }
        self.export_dir = "exports"
        os.makedirs(self.export_dir, exist_ok=True)

    def initialize_state(self):
        """Initialize export manager state"""
        if 'export_manager' not in st.session_state:
            st.session_state.export_manager = {
                'recording': False,
                'frames': [],
                'frame_rate': 30,
                'resolution': (1920, 1080),
                'quality': 'high',
                'format': 'MP4',
                'compression': 0.9,
                'include_metrics': True
            }

    def render_export_controls(self):
        """Render export control panel"""
        st.markdown("### Export Settings")
        
        # Export format selection
        format_type = st.selectbox(
            "Export Type",
            ['Video', 'Image', 'Data'],
            key='export_format_type'
        )
        
        formats = self.supported_formats[format_type.lower()]
        st.session_state.export_manager['format'] = st.selectbox(
            f"{format_type} Format",
            formats,
            key='export_format'
        )
        
        # Quality settings
        col1, col2 = st.columns(2)
        
        with col1:
            if format_type in ['Video', 'Image']:
                st.session_state.export_manager['quality'] = st.selectbox(
                    "Quality",
                    ['Low', 'Medium', 'High', 'Ultra'],
                    index=2,
                    key='export_quality'
                )
        
        with col2:
            if format_type == 'Video':
                st.session_state.export_manager['frame_rate'] = st.number_input(
                    "Frame Rate",
                    min_value=1,
                    max_value=60,
                    value=30,
                    key='export_frame_rate'
                )
        
        # Resolution settings
        if format_type in ['Video', 'Image']:
            st.markdown("#### Resolution")
            col1, col2 = st.columns(2)
            
            with col1:
                width = st.number_input(
                    "Width",
                    min_value=480,
                    max_value=3840,
                    value=1920,
                    step=160,
                    key='export_width'
                )
            
            with col2:
                height = st.number_input(
                    "Height",
                    min_value=360,
                    max_value=2160,
                    value=1080,
                    step=90,
                    key='export_height'
                )
            
            st.session_state.export_manager['resolution'] = (width, height)
        
        # Additional options
        st.markdown("#### Options")
        
        st.session_state.export_manager['include_metrics'] = st.checkbox(
            "Include Metrics",
            value=True,
            key='export_include_metrics'
        )
        
        if format_type in ['Video', 'Image']:
            st.session_state.export_manager['compression'] = st.slider(
                "Compression",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.1,
                key='export_compression'
            )
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if format_type == 'Video':
                if not st.session_state.export_manager['recording']:
                    if st.button("🔴 Start Recording"):
                        self.start_recording()
                else:
                    if st.button("⏹️ Stop Recording"):
                        self.stop_recording()
        
        with col2:
            if st.button("💾 Export Current View"):
                self.export_current_view()

    def start_recording(self):
        """Start recording frames"""
        st.session_state.export_manager['recording'] = True
        st.session_state.export_manager['frames'] = []

    def stop_recording(self):
        """Stop recording and save video"""
        st.session_state.export_manager['recording'] = False
        if len(st.session_state.export_manager['frames']) > 0:
            self.save_video()

    def add_frame(self, frame: np.ndarray, metrics: Optional[Dict] = None):
        """Add a frame to the recording buffer"""
        if st.session_state.export_manager['recording']:
            # Resize frame
            target_size = st.session_state.export_manager['resolution']
            frame = cv2.resize(frame, target_size)
            
            # Add metrics overlay if enabled
            if st.session_state.export_manager['include_metrics'] and metrics is not None:
                frame = self._add_metrics_overlay(frame, metrics)
            
            st.session_state.export_manager['frames'].append(frame)

    def export_current_view(self):
        """Export the current view"""
        state = st.session_state.export_manager
        format_type = st.session_state.export_format_type.lower()
        
        if format_type == 'image':
            self._export_image()
        elif format_type == 'data':
            self._export_data()
        else:
            st.error("Please use the recording feature for video export")

    def save_video(self):
        """Save recorded video"""
        state = st.session_state.export_manager
        frames = state['frames']
        
        if len(frames) == 0:
            st.error("No frames to export")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.export_dir, f"simulation_{timestamp}")
        
        format_type = state['format']
        if format_type == 'MP4':
            self._save_mp4(filename, frames)
        elif format_type == 'GIF':
            self._save_gif(filename, frames)
        elif format_type == 'AVI':
            self._save_avi(filename, frames)
        
        # Clear frame buffer
        state['frames'] = []
        
        st.success(f"Video saved as {filename}.{format_type.lower()}")

    def _save_mp4(self, filename: str, frames: List[np.ndarray]):
        """Save frames as MP4 video"""
        fps = st.session_state.export_manager['frame_rate']
        height, width = frames[0].shape[:2]
        
        writer = cv2.VideoWriter(
            f"{filename}.mp4",
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        writer.release()

    def _save_gif(self, filename: str, frames: List[np.ndarray]):
        """Save frames as GIF"""
        fps = st.session_state.export_manager['frame_rate']
        duration = 1000 / fps  # Duration per frame in milliseconds
        
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            f"{filename}.gif",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

    def _save_avi(self, filename: str, frames: List[np.ndarray]):
        """Save frames as AVI video"""
        fps = st.session_state.export_manager['frame_rate']
        height, width = frames[0].shape[:2]
        
        writer = cv2.VideoWriter(
            f"{filename}.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        writer.release()

    def _export_image(self):
        """Export current view as image"""
        state = st.session_state.export_manager
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.export_dir, f"snapshot_{timestamp}")
        
        # Get current figure
        fig = plt.gcf()
        
        # Set figure size based on resolution
        width, height = state['resolution']
        fig.set_size_inches(width/100, height/100)
        
        # Save with appropriate format
        format_type = state['format'].lower()
        dpi = self._get_dpi_for_quality(state['quality'])
        
        plt.savefig(
            f"{filename}.{format_type}",
            format=format_type,
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
            quality=int(state['compression'] * 100) if format_type == 'jpg' else None
        )
        
        st.success(f"Image saved as {filename}.{format_type}")

    def _export_data(self):
        """Export simulation data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.export_dir, f"data_{timestamp}")
        
        # Get current simulation state
        state = st.session_state.solver.get_state()
        format_type = st.session_state.export_manager['format'].lower()
        
        if format_type == 'npz':
            np.savez(f"{filename}.npz", **state)
        elif format_type == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_state = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in state.items()
            }
            with open(f"{filename}.json", 'w') as f:
                json.dump(json_state, f, indent=4)
        elif format_type == 'csv':
            # Save each array as a separate CSV file
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    np.savetxt(
                        f"{filename}_{key}.csv",
                        value,
                        delimiter=',',
                        header=f"{key} data"
                    )
        
        st.success(f"Data exported to {filename}.{format_type}")

    def _add_metrics_overlay(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Add metrics overlay to frame"""
        # Create a semi-transparent overlay
        overlay = frame.copy()
        alpha = 0.8
        
        # Add dark background for text
        cv2.rectangle(
            overlay,
            (10, 10),
            (300, 120),
            (0, 0, 0),
            cv2.FILLED
        )
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add metrics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        for key, value in metrics.items():
            text = f"{key}: {value:.2f}"
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                font,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 25
        
        return frame

    def _get_dpi_for_quality(self, quality: str) -> int:
        """Get DPI value for given quality setting"""
        quality_map = {
            'Low': 72,
            'Medium': 150,
            'High': 300,
            'Ultra': 600
        }
        return quality_map.get(quality, 300)

    def get_state(self) -> Dict:
        """Get current export manager state"""
        return st.session_state.export_manager 