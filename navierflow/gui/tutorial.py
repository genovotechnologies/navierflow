"""
Tutorial and Help System for NavierFlow
"""
from enum import Enum
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


class TutorialStep(Enum):
    """Tutorial step identifiers"""
    WELCOME = "welcome"
    BASIC_CONTROLS = "basic_controls"
    VISUALIZATION_MODES = "visualization_modes"
    SIMULATION_SETTINGS = "simulation_settings"
    ADVANCED_FEATURES = "advanced_features"
    EXPORT_DATA = "export_data"
    ANALYTICS = "analytics"
    COMPLETE = "complete"


@dataclass
class TutorialContent:
    """Content for a tutorial step"""
    title: str
    description: str
    instructions: List[str]
    tips: Optional[List[str]] = None
    image_path: Optional[str] = None


class TutorialSystem:
    """Interactive tutorial system"""
    
    def __init__(self):
        self.current_step = 0
        self.steps: List[TutorialContent] = []
        self.completed_steps = set()
        self.is_active = False
        self._initialize_tutorials()
    
    def _initialize_tutorials(self):
        """Initialize tutorial content"""
        self.steps = [
            TutorialContent(
                title="Welcome to NavierFlow",
                description="NavierFlow is an advanced fluid dynamics simulation engine. This tutorial will guide you through the basics.",
                instructions=[
                    "Click 'Next' to continue to the next step",
                    "You can skip the tutorial at any time by clicking 'Skip'",
                    "Access this tutorial again from the Help menu"
                ],
                tips=[
                    "Take your time to explore each feature",
                    "Experiment with different settings"
                ]
            ),
            TutorialContent(
                title="Basic Controls",
                description="Learn how to interact with the simulation.",
                instructions=[
                    "Left-click and drag in the visualization area to interact with the fluid",
                    "Use the mouse wheel to zoom in/out (3D mode)",
                    "Right-click and drag to rotate the view (3D mode)",
                    "Use the control panel on the left to adjust parameters"
                ],
                tips=[
                    "The brush size slider controls the interaction area",
                    "Try different force intensities for varied effects"
                ]
            ),
            TutorialContent(
                title="Visualization Modes",
                description="NavierFlow offers multiple visualization modes.",
                instructions=[
                    "Use the 'Visualization Type' dropdown to switch modes",
                    "Try Surface mode for height field visualization",
                    "Volume mode shows 3D density distribution",
                    "Streamline mode displays flow paths",
                    "Isosurface mode shows constant-value surfaces"
                ],
                tips=[
                    "Each mode highlights different aspects of the simulation",
                    "Combine with color schemes for better visualization"
                ]
            ),
            TutorialContent(
                title="Simulation Settings",
                description="Customize the simulation parameters.",
                instructions=[
                    "Adjust viscosity to change fluid thickness",
                    "Modify time step for simulation speed/accuracy",
                    "Select different physics models for various scenarios",
                    "Change numerical methods for different accuracy levels"
                ],
                tips=[
                    "Lower viscosity = more chaotic flow",
                    "Smaller time steps = more accurate but slower",
                    "Experiment with presets for quick setup"
                ]
            ),
            TutorialContent(
                title="Advanced Features",
                description="Explore advanced simulation features.",
                instructions=[
                    "Enable turbulence modeling for complex flows",
                    "Use multi-phase flow for liquid-gas interactions",
                    "Add heat transfer for thermal simulations",
                    "Enable AI enhancement for optimized simulations"
                ],
                tips=[
                    "Advanced features require more computational power",
                    "Check the analytics panel for performance metrics"
                ]
            ),
            TutorialContent(
                title="Export and Analysis",
                description="Save and analyze your simulation data.",
                instructions=[
                    "Use File > Export to save simulation data",
                    "Choose format: VTK, CSV, or images",
                    "Enable recording for video capture",
                    "View real-time analytics in the dashboard"
                ],
                tips=[
                    "Export regularly to save progress",
                    "Use analytics to validate simulation accuracy"
                ]
            ),
            TutorialContent(
                title="Analytics Dashboard",
                description="Monitor simulation performance and metrics.",
                instructions=[
                    "Open the Analytics tab to view real-time metrics",
                    "Monitor FPS, memory usage, and GPU utilization",
                    "Track Reynolds number and flow characteristics",
                    "Use graphs to visualize temporal evolution"
                ],
                tips=[
                    "Performance metrics help optimize settings",
                    "Reynolds number indicates flow regime"
                ]
            ),
            TutorialContent(
                title="Tutorial Complete!",
                description="You've completed the NavierFlow tutorial.",
                instructions=[
                    "You're now ready to start simulating",
                    "Explore the examples in the File menu",
                    "Access documentation from Help > Documentation",
                    "Join our community for support and updates"
                ],
                tips=[
                    "Practice with different scenarios",
                    "Share your simulations with the community"
                ]
            )
        ]
    
    def start(self):
        """Start the tutorial"""
        self.is_active = True
        self.current_step = 0
        self.completed_steps.clear()
    
    def next_step(self) -> Optional[TutorialContent]:
        """Move to next tutorial step"""
        if self.current_step < len(self.steps):
            self.completed_steps.add(self.current_step)
            self.current_step += 1
            if self.current_step < len(self.steps):
                return self.steps[self.current_step]
            else:
                self.is_active = False
                return None
        return None
    
    def previous_step(self) -> Optional[TutorialContent]:
        """Move to previous tutorial step"""
        if self.current_step > 0:
            self.current_step -= 1
            return self.steps[self.current_step]
        return None
    
    def get_current_step(self) -> Optional[TutorialContent]:
        """Get current tutorial step"""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def skip(self):
        """Skip the tutorial"""
        self.is_active = False
        self.completed_steps.add(self.current_step)
    
    def restart(self):
        """Restart the tutorial"""
        self.start()
    
    def get_progress(self) -> tuple:
        """Get tutorial progress"""
        return (self.current_step + 1, len(self.steps))


class HelpSystem:
    """Help documentation system"""
    
    def __init__(self):
        self.topics: Dict[str, Dict] = {}
        self._initialize_help()
    
    def _initialize_help(self):
        """Initialize help topics"""
        self.topics = {
            "getting_started": {
                "title": "Getting Started",
                "content": """
# Getting Started with NavierFlow

NavierFlow is a professional-grade CFD simulation engine designed for both 
educational and research applications.

## Quick Start

1. Launch the application
2. Select a simulation preset or create a new one
3. Adjust parameters in the control panel
4. Click Play to start the simulation
5. Interact with the fluid by clicking and dragging

## Basic Concepts

- **Viscosity**: Controls fluid thickness (higher = thicker)
- **Reynolds Number**: Indicates flow regime (laminar vs turbulent)
- **Time Step**: Affects simulation speed and accuracy
- **Resolution**: Grid density for calculations
                """,
                "keywords": ["start", "begin", "introduction", "basics"]
            },
            "visualization": {
                "title": "Visualization Modes",
                "content": """
# Visualization Modes

NavierFlow offers multiple visualization modes to help you understand 
the flow behavior:

## Available Modes

### Surface Plot
Shows the scalar field as a 3D surface. Good for visualizing pressure 
or density distributions.

### Volume Rendering
Displays the full 3D volumetric data with transparency. Best for 
understanding internal flow structures.

### Streamlines
Shows the path that particles would follow in the flow. Excellent for 
visualizing flow direction and vortices.

### Isosurfaces
Displays surfaces of constant value. Useful for identifying regions 
with specific properties.

### Point Cloud
Represents discrete data points. Good for particle tracking and 
sparse data visualization.

## Tips

- Switch between modes to gain different insights
- Combine with color schemes for better visualization
- Use transparency settings for volume rendering
                """,
                "keywords": ["visual", "display", "render", "view", "3d", "2d"]
            },
            "simulation_settings": {
                "title": "Simulation Settings",
                "content": """
# Simulation Settings

Configure your simulation parameters for optimal results.

## Physics Parameters

### Viscosity
Controls how resistant the fluid is to flow. Higher values create 
thicker, more viscous fluids (like honey), while lower values create 
thinner fluids (like water).

### Density
Mass per unit volume of the fluid. Affects momentum and inertia.

### Time Step
Smaller time steps provide more accuracy but slower simulation. 
Larger time steps are faster but may be unstable.

## Numerical Methods

### Finite Volume
Conservative method, good for general CFD applications.

### Finite Element
Flexible for complex geometries, better for structural interactions.

### Lattice Boltzmann
Efficient for parallel computing, excellent for complex boundaries.

## Boundary Conditions

- **No-slip**: Fluid velocity is zero at walls
- **Free-slip**: No friction at walls
- **Inlet**: Specify inflow conditions
- **Outlet**: Specify outflow conditions
                """,
                "keywords": ["settings", "parameters", "configuration", "physics"]
            },
            "analytics": {
                "title": "Analytics and Monitoring",
                "content": """
# Analytics and Monitoring

Track simulation performance and physical properties.

## Performance Metrics

### FPS (Frames Per Second)
Indicates how fast the simulation is running. Higher is better.

### Memory Usage
Shows RAM consumption. Keep below 80% for optimal performance.

### GPU Utilization
Percentage of GPU being used. Higher values indicate better use of 
available hardware.

## Physical Metrics

### Reynolds Number
Re = ρvL/μ
- Re < 2300: Laminar flow
- Re > 4000: Turbulent flow
- Between: Transitional flow

### Average Velocity
Mean fluid velocity across the domain.

### Maximum Pressure
Peak pressure value in the simulation.

## Monitoring Tips

- Watch for sudden FPS drops (may indicate instability)
- High memory usage may require reducing resolution
- Use analytics to validate simulation accuracy
                """,
                "keywords": ["analytics", "performance", "metrics", "monitoring"]
            },
            "export": {
                "title": "Exporting Data",
                "content": """
# Exporting Data

Save your simulation results for further analysis.

## Export Formats

### VTK (ParaView)
Standard format for visualization in ParaView. Includes full field data.

### CSV
Comma-separated values for spreadsheet analysis.

### Images/Video
PNG sequences or MP4 video for presentations.

### HDF5
Efficient binary format for large datasets.

## Export Options

1. File > Export
2. Select format
3. Choose data to export (velocity, pressure, etc.)
4. Set export frequency
5. Specify output location

## Tips

- Export regularly to avoid data loss
- Use VTK for detailed post-processing
- Video export is great for presentations
- CSV is best for statistical analysis
                """,
                "keywords": ["export", "save", "output", "file"]
            },
            "troubleshooting": {
                "title": "Troubleshooting",
                "content": """
# Troubleshooting

Common issues and solutions.

## Simulation Issues

### Simulation is Unstable
- Reduce time step
- Increase viscosity
- Check boundary conditions
- Lower Reynolds number

### Slow Performance
- Reduce grid resolution
- Disable advanced features temporarily
- Close other applications
- Update graphics drivers

### Visualization Problems
- Try different visualization modes
- Adjust color scale range
- Check data validity
- Reset camera view

## Common Errors

### Out of Memory
- Reduce domain size
- Lower resolution
- Enable compression
- Close other applications

### GPU Errors
- Update graphics drivers
- Check GPU compatibility
- Reduce visual effects
- Try CPU mode

## Getting Help

- Check documentation
- Visit community forums
- Contact support
- Report bugs on GitHub
                """,
                "keywords": ["problem", "issue", "error", "fix", "help"]
            }
        }
    
    def get_topic(self, topic_id: str) -> Optional[Dict]:
        """Get help topic by ID"""
        return self.topics.get(topic_id)
    
    def search(self, query: str) -> List[Dict]:
        """Search help topics"""
        query = query.lower()
        results = []
        
        for topic_id, topic in self.topics.items():
            # Search in title, content, and keywords
            if (query in topic["title"].lower() or
                query in topic["content"].lower() or
                any(query in keyword for keyword in topic["keywords"])):
                results.append({
                    "id": topic_id,
                    "title": topic["title"],
                    "relevance": self._calculate_relevance(query, topic)
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results
    
    def _calculate_relevance(self, query: str, topic: Dict) -> float:
        """Calculate search relevance score"""
        score = 0.0
        
        # Title match
        if query in topic["title"].lower():
            score += 10.0
        
        # Keyword match
        for keyword in topic["keywords"]:
            if query in keyword:
                score += 5.0
        
        # Content match
        content_lower = topic["content"].lower()
        score += content_lower.count(query) * 0.5
        
        return score
    
    def get_all_topics(self) -> List[str]:
        """Get list of all topic IDs"""
        return list(self.topics.keys())
