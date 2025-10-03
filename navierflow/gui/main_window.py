import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QToolBar,
    QMenu,
    QMenuBar,
    QGroupBox,
    QScrollArea,
    QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QImage, QPixmap, QIcon, QAction, QPalette, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from .animations import EasingFunctions, Animator, TransitionManager, ParticleSystem

class VisualizationType(Enum):
    """Types of visualizations"""
    SURFACE = "Surface"
    VOLUME = "Volume"
    STREAMLINE = "Streamline"
    ISOSURFACE = "Isosurface"
    POINT_CLOUD = "Point Cloud"

@dataclass
class GUIConfig:
    """GUI configuration"""
    window_size: Tuple[int, int] = (1200, 800)
    update_interval: int = 100  # ms
    use_3d: bool = True
    use_antialiasing: bool = True
    use_shadows: bool = True
    use_ambient_occlusion: bool = True
    use_volumetric_lighting: bool = True
    use_depth_of_field: bool = True
    use_motion_blur: bool = True
    colormap: str = "viridis"
    background_color: str = "#1e1e1e"
    text_color: str = "#ffffff"
    accent_color: str = "#007acc"
    font_size: int = 12
    show_axes: bool = True
    show_grid: bool = True
    show_legend: bool = True
    show_colorbar: bool = True
    animation_fps: int = 30
    animation_duration: float = 10.0

class MainWindow(QMainWindow):
    def __init__(self, config: Optional[GUIConfig] = None):
        """
        Initialize main window
        
        Args:
            config: GUI configuration
        """
        super().__init__()
        self.config = config or GUIConfig()
        self.animation_time = 0.0
        self.transition_manager = TransitionManager()
        self.particle_system = ParticleSystem(max_particles=500)
        
        # Initialize animators for smooth transitions
        self.transition_manager.add_animation('camera_distance', Animator(1.0, EasingFunctions.ease_in_out_cubic))
        self.transition_manager.add_animation('rotation', Animator(0.5, EasingFunctions.ease_in_out_sine))
        
        self.setup_ui()
        self._setup_visualization()
        self._setup_timer()
        
    def setup_ui(self):
        """Setup user interface"""
        # Set window properties
        self.setWindowTitle("NavierFlow - Advanced Fluid Dynamics Simulator")
        self.resize(*self.config.window_size)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Create control panel
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Create visualization panel
        visualization_panel = self._create_visualization_panel()
        splitter.addWidget(visualization_panel)
        
        # Set splitter sizes
        splitter.setSizes([300, 900])
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        new_action = QAction("New Simulation", self)
        open_action = QAction("Open", self)
        save_action = QAction("Save", self)
        export_action = QAction("Export", self)
        exit_action = QAction("Exit", self)
        
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        preferences_action = QAction("Preferences", self)
        edit_menu.addAction(preferences_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        reset_view_action = QAction("Reset View", self)
        view_menu.addAction(reset_view_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        documentation_action = QAction("Documentation", self)
        help_menu.addAction(about_action)
        help_menu.addAction(documentation_action)
        
    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add toolbar actions
        play_action = QAction("Play", self)
        pause_action = QAction("Pause", self)
        stop_action = QAction("Stop", self)
        reset_action = QAction("Reset", self)
        
        toolbar.addAction(play_action)
        toolbar.addAction(pause_action)
        toolbar.addAction(stop_action)
        toolbar.addAction(reset_action)
        
    def create_control_panel(self) -> QWidget:
        """Create control panel"""
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll)
        
        # Create content widget
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        
        # Simulation settings group
        sim_group = QGroupBox("Simulation Settings")
        sim_layout = QVBoxLayout(sim_group)
        
        # Physics model selection
        model_label = QLabel("Physics Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Incompressible Flow",
            "Compressible Flow",
            "Multiphase Flow",
            "Heat Transfer",
            "Mass Transfer"
        ])
        sim_layout.addWidget(model_label)
        sim_layout.addWidget(self.model_combo)
        
        # Numerical method selection
        method_label = QLabel("Numerical Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Finite Volume",
            "Finite Element",
            "Finite Difference",
            "Lattice Boltzmann"
        ])
        sim_layout.addWidget(method_label)
        sim_layout.addWidget(self.method_combo)
        
        # Time step control
        timestep_label = QLabel("Time Step:")
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.0001, 1.0)
        self.timestep_spin.setValue(0.001)
        self.timestep_spin.setSingleStep(0.0001)
        sim_layout.addWidget(timestep_label)
        sim_layout.addWidget(self.timestep_spin)
        
        # Maximum steps control
        maxsteps_label = QLabel("Maximum Steps:")
        self.maxsteps_spin = QSpinBox()
        self.maxsteps_spin.setRange(1, 1000000)
        self.maxsteps_spin.setValue(1000)
        sim_layout.addWidget(maxsteps_label)
        sim_layout.addWidget(self.maxsteps_spin)
        
        time_step_layout = QHBoxLayout()
        time_step_layout.addWidget(QLabel("Time Step:"))
        self.time_step_spin = QDoubleSpinBox()
        self.time_step_spin.setRange(0.001, 1.0)
        self.time_step_spin.setValue(0.01)
        self.time_step_spin.setSingleStep(0.001)
        time_step_layout.addWidget(self.time_step_spin)
        simulation_layout.addLayout(time_step_layout)
        
        # Simulation speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(1)
        speed_layout.addWidget(self.speed_spin)
        simulation_layout.addLayout(speed_layout)
        
        # Simulation control buttons
        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_button)
        simulation_layout.addLayout(button_layout)
        
        # Add simulation tab
        tabs.addTab(simulation_tab, "Simulation")
        
        # Visualization tab
        visualization_tab = QWidget()
        visualization_layout = QVBoxLayout(visualization_tab)
        
        # Visualization type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([t.value for t in VisualizationType])
        type_layout.addWidget(self.type_combo)
        visualization_layout.addLayout(type_layout)
        
        # Visualization options
        self.antialiasing_check = QCheckBox("Antialiasing")
        self.antialiasing_check.setChecked(self.config.use_antialiasing)
        visualization_layout.addWidget(self.antialiasing_check)
        
        self.shadows_check = QCheckBox("Shadows")
        self.shadows_check.setChecked(self.config.use_shadows)
        visualization_layout.addWidget(self.shadows_check)
        
        self.ao_check = QCheckBox("Ambient Occlusion")
        self.ao_check.setChecked(self.config.use_ambient_occlusion)
        visualization_layout.addWidget(self.ao_check)
        
        self.volumetric_check = QCheckBox("Volumetric Lighting")
        self.volumetric_check.setChecked(self.config.use_volumetric_lighting)
        visualization_layout.addWidget(self.volumetric_check)
        
        self.dof_check = QCheckBox("Depth of Field")
        self.dof_check.setChecked(self.config.use_depth_of_field)
        visualization_layout.addWidget(self.dof_check)
        
        self.motion_blur_check = QCheckBox("Motion Blur")
        self.motion_blur_check.setChecked(self.config.use_motion_blur)
        visualization_layout.addWidget(self.motion_blur_check)
        
        # Add visualization tab
        tabs.addTab(visualization_tab, "Visualization")
        
        # Add tabs to layout
        layout.addWidget(tabs)
        
        return panel
        
    def _create_visualization_panel(self) -> QWidget:
        """Create visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        if self.config.use_3d:
            # Create 3D visualization
            self.view = gl.GLViewWidget()
            self.view.setCameraPosition(distance=10)
            
            # Add grid
            grid = gl.GLGridItem()
            self.view.addItem(grid)
            
            # Add visualization to layout
            layout.addWidget(self.view)
        else:
            # Create 2D visualization
            self.plot = pg.PlotWidget()
            self.plot.showGrid(True, True)
            
            # Add visualization to layout
            layout.addWidget(self.plot)
            
        return panel
        
    def _setup_visualization(self):
        """Setup visualization"""
        if self.config.use_3d:
            # Create surface plot
            self.surface = gl.GLMeshItem(
                smooth=True,
                drawEdges=False,
                drawFaces=True,
                shader="shaded"
            )
            self.view.addItem(self.surface)
            
            # Create volume plot
            self.volume = gl.GLVolumeItem(
                data=np.zeros((100, 100, 100)),
                smooth=True
            )
            self.view.addItem(self.volume)
            
            # Create streamline plot
            self.streamlines = gl.GLLinePlotItem(
                pos=np.zeros((100, 3)),
                color=(1, 1, 1, 1),
                width=2
            )
            self.view.addItem(self.streamlines)
            
            # Create isosurface plot
            self.isosurface = gl.GLMeshItem(
                smooth=True,
                drawEdges=False,
                drawFaces=True,
                shader="shaded"
            )
            self.view.addItem(self.isosurface)
            
            # Create point cloud plot
            self.point_cloud = gl.GLScatterPlotItem(
                pos=np.zeros((100, 3)),
                color=(1, 1, 1, 1),
                size=5
            )
            self.view.addItem(self.point_cloud)
        else:
            # Create 2D plots
            self.surface_plot = self.plot.plot(pen="b")
            self.volume_plot = self.plot.plot(pen="r")
            self.streamline_plot = self.plot.plot(pen="g")
            self.isosurface_plot = self.plot.plot(pen="y")
            self.point_cloud_plot = self.plot.plot(pen="w", symbol="o")
            
    def _setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(self.config.update_interval)
        self._start_time = 0
        
    def _update(self):
        """Update visualization"""
        # Update animation time
        self.animation_time += self.config.update_interval / 1000.0
        
        # Update transitions
        transitions = self.transition_manager.update(self.animation_time)
        
        # Update particle system
        self.particle_system.update(self.config.update_interval / 1000.0)
        
        # Get current visualization type
        viz_type = VisualizationType(self.type_combo.currentText())
        
        # Update visualization based on type
        if self.config.use_3d:
            if viz_type == VisualizationType.SURFACE:
                self._update_surface_3d()
            elif viz_type == VisualizationType.VOLUME:
                self._update_volume_3d()
            elif viz_type == VisualizationType.STREAMLINE:
                self._update_streamline_3d()
            elif viz_type == VisualizationType.ISOSURFACE:
                self._update_isosurface_3d()
            elif viz_type == VisualizationType.POINT_CLOUD:
                self._update_point_cloud_3d()
        else:
            if viz_type == VisualizationType.SURFACE:
                self._update_surface_2d()
            elif viz_type == VisualizationType.VOLUME:
                self._update_volume_2d()
            elif viz_type == VisualizationType.STREAMLINE:
                self._update_streamline_2d()
            elif viz_type == VisualizationType.ISOSURFACE:
                self._update_isosurface_2d()
            elif viz_type == VisualizationType.POINT_CLOUD:
                self._update_point_cloud_2d()
                
    def _update_surface_3d(self):
        """Update 3D surface visualization"""
        # Generate test data for surface with smooth animation
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2) + self.animation_time * 2) * np.exp(-np.sqrt(X**2 + Y**2) / 10)
        
        # Update mesh data with smooth animation
        vertices = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        faces = []
        for i in range(len(x)-1):
            for j in range(len(y)-1):
                idx = i * len(y) + j
                faces.append([idx, idx+1, idx+len(y)])
                faces.append([idx+1, idx+len(y)+1, idx+len(y)])
        
        self.surface.setMeshData(vertexes=vertices, faces=np.array(faces))
        
    def _update_volume_3d(self):
        """Update 3D volume visualization"""
        # Generate test volumetric data with animation
        t = self.animation_time
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        z = np.linspace(-2, 2, 50)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        data = np.sin(X + t) * np.cos(Y + t) * np.sin(Z + t) * np.exp(-(X**2 + Y**2 + Z**2) / 10)
        
        self.volume.setData(data)
        
    def _update_streamline_3d(self):
        """Update 3D streamline visualization"""
        # Generate test streamline data with smooth animation
        t = self.animation_time
        n_points = 100
        theta = np.linspace(0, 4*np.pi, n_points)
        r = np.linspace(0, 5, n_points)
        x = r * np.cos(theta + t)
        y = r * np.sin(theta + t)
        z = np.sin(theta * 2 + t) * 2
        
        pos = np.column_stack([x, y, z])
        colors = np.zeros((n_points, 4))
        colors[:, 0] = (np.sin(theta + t) + 1) / 2  # R
        colors[:, 1] = (np.cos(theta + t) + 1) / 2  # G
        colors[:, 2] = 0.5  # B
        colors[:, 3] = 0.8  # Alpha
        
        self.streamlines.setData(pos=pos, color=colors)
        
    def _update_isosurface_3d(self):
        """Update 3D isosurface visualization"""
        # Generate isosurface data with animation
        t = self.animation_time
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        z = np.linspace(-3, 3, 30)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        data = X**2 + Y**2 + Z**2 - (4 + 2 * np.sin(t))
        
        # Simple isosurface extraction at level 0
        vertices, faces = self._extract_isosurface(data, level=0.0)
        if len(vertices) > 0:
            self.isosurface.setMeshData(vertexes=vertices, faces=faces)
    
    def _extract_isosurface(self, data, level=0.0):
        """Extract isosurface using marching cubes approximation"""
        try:
            from skimage import measure
            verts, faces, _, _ = measure.marching_cubes(data, level=level)
            return verts, faces
        except:
            # Return empty if extraction fails
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        
    def _update_point_cloud_3d(self):
        """Update 3D point cloud visualization"""
        # Generate test point cloud data with animation
        t = self.animation_time
        n_points = 1000
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = 5 + np.sin(t * 2) * 2
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        pos = np.column_stack([x, y, z])
        
        # Animated colors
        colors = np.zeros((n_points, 4))
        colors[:, 0] = (np.sin(theta + t) + 1) / 2  # R
        colors[:, 1] = (np.cos(phi + t) + 1) / 2    # G
        colors[:, 2] = (np.sin(t) + 1) / 2          # B
        colors[:, 3] = 0.8  # Alpha
        
        self.point_cloud.setData(pos=pos, color=colors)
        
    def _update_surface_2d(self):
        """Update 2D surface visualization"""
        # Generate test data for 2D surface with animation
        t = self.animation_time
        x = np.linspace(0, 10, 100)
        y = np.sin(x + t) * np.exp(-x/10)
        
        self.surface_plot.setData(x, y)
        
    def _update_volume_2d(self):
        """Update 2D volume visualization"""
        # Generate test data for 2D volume (cross-section) with animation
        t = self.animation_time
        x = np.linspace(0, 10, 100)
        y = np.cos(x + t * 1.5) * np.exp(-x/10)
        
        self.volume_plot.setData(x, y)
        
    def _update_streamline_2d(self):
        """Update 2D streamline visualization"""
        # Generate test streamline data with animation
        t = self.animation_time
        x = np.linspace(0, 10, 100)
        y = np.sin(2*x + t) * (1 + 0.3 * np.cos(t))
        
        self.streamline_plot.setData(x, y)
        
    def _update_isosurface_2d(self):
        """Update 2D isosurface visualization"""
        # Generate test isosurface data (contour) with animation
        t = self.animation_time
        x = np.linspace(0, 10, 100)
        y = np.sin(3*x + t * 2) * 0.5 * (1 + 0.2 * np.sin(t))
        
        self.isosurface_plot.setData(x, y)
        
    def _update_point_cloud_2d(self):
        """Update 2D point cloud visualization"""
        # Generate test point cloud data with animation
        t = self.animation_time
        n_points = 100
        x = np.random.uniform(0, 10, n_points)
        y = np.sin(x + t) + np.random.normal(0, 0.1, n_points)
        
        self.point_cloud_plot.setData(x, y)
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.timer.stop()
        event.accept()

def main():
    """Main function"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 