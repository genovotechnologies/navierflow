"""
Data export functionality for NavierFlow
"""
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import json


class ExportFormat(Enum):
    """Supported export formats"""
    VTK = "vtk"
    CSV = "csv"
    PNG = "png"
    JPEG = "jpeg"
    MP4 = "mp4"
    GIF = "gif"
    HDF5 = "hdf5"
    JSON = "json"
    NUMPY = "npy"


class DataExporter:
    """Export simulation data in various formats"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_vtk(self, data: Dict[str, np.ndarray], filename: str,
                   grid_shape: Tuple[int, int, int] = None):
        """
        Export data in VTK format for ParaView
        
        Args:
            data: Dictionary of field data (velocity, pressure, etc.)
            filename: Output filename
            grid_shape: Grid dimensions
        """
        filepath = self.output_dir / f"{filename}.vtk"
        
        try:
            import vtk
            from vtk.util import numpy_support
            
            # Create structured grid
            if grid_shape is None:
                grid_shape = data[list(data.keys())[0]].shape[:3]
            
            grid = vtk.vtkStructuredGrid()
            grid.SetDimensions(*grid_shape)
            
            # Create points
            points = vtk.vtkPoints()
            for k in range(grid_shape[2]):
                for j in range(grid_shape[1]):
                    for i in range(grid_shape[0]):
                        points.InsertNextPoint(i, j, k)
            grid.SetPoints(points)
            
            # Add field data
            for field_name, field_data in data.items():
                vtk_array = numpy_support.numpy_to_vtk(field_data.ravel(), deep=True)
                vtk_array.SetName(field_name)
                grid.GetPointData().AddArray(vtk_array)
            
            # Write to file
            writer = vtk.vtkStructuredGridWriter()
            writer.SetFileName(str(filepath))
            writer.SetInputData(grid)
            writer.Write()
            
            return str(filepath)
        except ImportError:
            # Fallback: simple VTK ASCII format
            return self._export_vtk_ascii(data, filename, grid_shape)
    
    def _export_vtk_ascii(self, data: Dict[str, np.ndarray], filename: str,
                          grid_shape: Tuple[int, int, int]):
        """Export in simple ASCII VTK format"""
        filepath = self.output_dir / f"{filename}.vtk"
        
        if grid_shape is None:
            grid_shape = data[list(data.keys())[0]].shape[:3]
        
        with open(filepath, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write("NavierFlow Export\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n")
            
            # Points
            n_points = np.prod(grid_shape)
            f.write(f"POINTS {n_points} float\n")
            for k in range(grid_shape[2]):
                for j in range(grid_shape[1]):
                    for i in range(grid_shape[0]):
                        f.write(f"{i} {j} {k}\n")
            
            # Point data
            f.write(f"POINT_DATA {n_points}\n")
            for field_name, field_data in data.items():
                if field_data.ndim == 4 and field_data.shape[-1] == 3:
                    # Vector field
                    f.write(f"VECTORS {field_name} float\n")
                    for k in range(grid_shape[2]):
                        for j in range(grid_shape[1]):
                            for i in range(grid_shape[0]):
                                v = field_data[i, j, k]
                                f.write(f"{v[0]} {v[1]} {v[2]}\n")
                else:
                    # Scalar field
                    f.write(f"SCALARS {field_name} float 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for k in range(grid_shape[2]):
                        for j in range(grid_shape[1]):
                            for i in range(grid_shape[0]):
                                f.write(f"{field_data[i, j, k]}\n")
        
        return str(filepath)
    
    def export_csv(self, data: Dict[str, np.ndarray], filename: str):
        """
        Export data in CSV format
        
        Args:
            data: Dictionary of field data
            filename: Output filename
        """
        filepath = self.output_dir / f"{filename}.csv"
        
        # Flatten all data
        rows = []
        field_names = list(data.keys())
        
        # Get dimensions
        first_field = data[field_names[0]]
        shape = first_field.shape
        
        # Create header
        header = ["x", "y", "z"] + field_names
        rows.append(header)
        
        # Add data rows
        if len(shape) == 3:
            # 3D scalar data
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        row = [i, j, k]
                        for field_name in field_names:
                            row.append(data[field_name][i, j, k])
                        rows.append(row)
        elif len(shape) == 4:
            # 3D vector data
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        row = [i, j, k]
                        for field_name in field_names:
                            values = data[field_name][i, j, k]
                            row.extend(values)
                        rows.append(row)
        
        # Write CSV
        with open(filepath, 'w') as f:
            for row in rows:
                f.write(','.join(map(str, row)) + '\n')
        
        return str(filepath)
    
    def export_numpy(self, data: Dict[str, np.ndarray], filename: str):
        """
        Export data in NumPy format
        
        Args:
            data: Dictionary of field data
            filename: Output filename
        """
        filepath = self.output_dir / f"{filename}.npz"
        np.savez_compressed(filepath, **data)
        return str(filepath)
    
    def export_json(self, data: Dict, filename: str):
        """
        Export metadata in JSON format
        
        Args:
            data: Dictionary of metadata
            filename: Output filename
        """
        filepath = self.output_dir / f"{filename}.json"
        
        # Convert numpy arrays to lists
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return str(filepath)
    
    def export_image(self, image_data: np.ndarray, filename: str,
                     format: ExportFormat = ExportFormat.PNG):
        """
        Export image
        
        Args:
            image_data: Image array (H, W, C)
            filename: Output filename
            format: Image format
        """
        try:
            from PIL import Image
            
            # Ensure data is in correct range
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            
            # Create image
            if image_data.ndim == 2:
                img = Image.fromarray(image_data, mode='L')
            elif image_data.shape[2] == 3:
                img = Image.fromarray(image_data, mode='RGB')
            elif image_data.shape[2] == 4:
                img = Image.fromarray(image_data, mode='RGBA')
            else:
                raise ValueError(f"Unsupported image shape: {image_data.shape}")
            
            # Save
            ext = format.value
            filepath = self.output_dir / f"{filename}.{ext}"
            img.save(filepath)
            
            return str(filepath)
        except ImportError:
            # Fallback: use matplotlib
            import matplotlib.pyplot as plt
            ext = format.value
            filepath = self.output_dir / f"{filename}.{ext}"
            plt.imsave(filepath, image_data)
            return str(filepath)
    
    def export_video(self, frames: List[np.ndarray], filename: str,
                     fps: int = 30, format: ExportFormat = ExportFormat.MP4):
        """
        Export video from frames
        
        Args:
            frames: List of frame arrays
            filename: Output filename
            fps: Frames per second
            format: Video format
        """
        try:
            import imageio
            
            ext = format.value
            filepath = self.output_dir / f"{filename}.{ext}"
            
            # Convert frames if needed
            processed_frames = []
            for frame in frames:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                processed_frames.append(frame)
            
            # Write video
            if format == ExportFormat.GIF:
                imageio.mimsave(filepath, processed_frames, fps=fps)
            else:
                imageio.mimsave(filepath, processed_frames, fps=fps, codec='libx264')
            
            return str(filepath)
        except ImportError:
            print("Warning: imageio not available for video export")
            # Fallback: save as image sequence
            return self.export_image_sequence(frames, filename)
    
    def export_image_sequence(self, frames: List[np.ndarray], filename: str):
        """
        Export frames as image sequence
        
        Args:
            frames: List of frame arrays
            filename: Base filename
        """
        filepaths = []
        for i, frame in enumerate(frames):
            frame_filename = f"{filename}_{i:06d}"
            filepath = self.export_image(frame, frame_filename)
            filepaths.append(filepath)
        
        return filepaths
    
    def export_hdf5(self, data: Dict[str, np.ndarray], filename: str,
                    compression: str = "gzip"):
        """
        Export data in HDF5 format
        
        Args:
            data: Dictionary of field data
            filename: Output filename
            compression: Compression method
        """
        try:
            import h5py
            
            filepath = self.output_dir / f"{filename}.h5"
            
            with h5py.File(filepath, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value, compression=compression)
            
            return str(filepath)
        except ImportError:
            print("Warning: h5py not available for HDF5 export")
            # Fallback to numpy
            return self.export_numpy(data, filename)


class SimulationRecorder:
    """Record simulation frames for video export"""
    
    def __init__(self, max_frames: int = 1000):
        """
        Initialize recorder
        
        Args:
            max_frames: Maximum number of frames to store
        """
        self.frames: List[np.ndarray] = []
        self.max_frames = max_frames
        self.is_recording = False
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.frames.clear()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the recording"""
        if self.is_recording:
            if len(self.frames) < self.max_frames:
                self.frames.append(frame.copy())
            else:
                print(f"Warning: Maximum frame count ({self.max_frames}) reached")
    
    def get_frames(self) -> List[np.ndarray]:
        """Get recorded frames"""
        return self.frames
    
    def clear(self):
        """Clear recorded frames"""
        self.frames.clear()
    
    def get_frame_count(self) -> int:
        """Get number of recorded frames"""
        return len(self.frames)
