
# Fluid Simulation

A Python-based interactive fluid simulation using OpenGL that supports both Eulerian and Lattice Boltzmann Method (LBM) approaches for fluid dynamics visualization.

![Language](https://img.shields.io/badge/language-Python-blue.svg)
[![Last Updated](https://img.shields.io/badge/last%20updated-2025--02--04-brightgreen.svg)](https://github.com/tafolabi009/fluidSimulation)

## Features

- Two simulation methods:
  - Eulerian fluid dynamics
  - Lattice Boltzmann Method (LBM)
- Interactive fluid manipulation with mouse
- Multiple visualization modes:
  - Blue gradient (default)
  - Fire gradient
  - Rainbow gradient
  - Grayscale
- Physics features:
  - Vorticity confinement
  - Temperature effects
  - Buoyancy
  - Interactive ball/particle physics
- Real-time performance monitoring
- Fullscreen support
- Customizable parameters

## Requirements

- Python 3.x
- NumPy
- OpenGL (PyOpenGL)
- GLUT

## Installation

```bash
pip install numpy PyOpenGL PyOpenGL-accelerate
```

## Usage

Run the simulation with default parameters:

```bash
python app.py
```

### Command Line Arguments

- `--method`: Choose simulation method (`eulerian` or `lbm`)
- `--size`: Set grid size (N x N)
- `--color-mode`: Set visualization mode (`blue`, `fire`, `rainbow`, `grayscale`)
- `--viscosity`: Set fluid viscosity
- `--vorticity`: Set vorticity confinement strength

Example:
```bash
python app.py --method lbm --size 256 --color-mode fire --viscosity 0.0001
```

## Controls

### Basic Controls
- Mouse drag: Add fluid
- F: Toggle fullscreen
- ESC: Exit fullscreen / Minimize window
- M: Switch method (Eulerian/LBM)
- R: Reset simulation
- C: Cycle color modes
- T: Toggle temperature effect
- V: Toggle vorticity confinement
- Arrow keys: Adjust brush size and viscosity
- H: Show help message

### Ball/Particle Controls
- Click and drag ball to move
- Page Up/Down: Resize ball
- I/K: Adjust ball interaction strength

## Technical Details

### Simulation Methods

1. **Eulerian Method**
   - Uses a grid-based approach
   - Implements velocity diffusion
   - Features pressure projection
   - Includes advection for density and velocity fields

2. **Lattice Boltzmann Method (LBM)**
   - Uses D2Q9 lattice model
   - Implements collision and streaming steps
   - Features bounce-back boundary conditions
   - Includes equilibrium distribution calculations

### Physics Features

- **Vorticity Confinement**: Helps preserve small-scale fluid features
- **Temperature Effects**: Influences fluid behavior through buoyancy
- **Interactive Ball Physics**: Implements two-way coupling between ball and fluid
- **Boundary Conditions**: Implements proper fluid boundaries and wall interactions

### Performance Optimization

- Uses NumPy for efficient array operations
- Implements mipmap generation for better rendering quality
- Features FPS monitoring and display
- Includes adaptive time-stepping

## Implementation Details

The simulation is built using:
- OpenGL for rendering
- GLUT for window management and user input
- NumPy for numerical computations
- Custom color gradient generators for visualization

### Key Components

1. **SimulationParams**: Data class containing all simulation parameters
   ```python
   @dataclass
   class SimulationParams:
       dt: float = 0.1
       viscosity: float = 0.0001
       omega: float = 1.0
       density_multiplier: float = 5.0
       velocity_multiplier: float = 2.0
       diffusion_rate: float = 0.05
       color_mode: str = 'blue'
       brush_size: int = 3
       vorticity: float = 0.1
       temperature: float = 0.0
       ball_radius: int = 10
       ball_interaction_strength: float = 3.0
   ```

2. **FluidSimulationOpenGL**: Main simulation class
   - Handles rendering
   - Processes user input
   - Implements both simulation methods
   - Manages state and parameters

## License

Open-source - License terms to be specified.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Repository Information

- **Repository**: [tafolabi009/fluidSimulation](https://github.com/tafolabi009/fluidSimulation)
- **Language**: 100% Python
- **Last Updated**: 2025-02-04

---
Created and maintained by [@tafolabi009](https://github.com/tafolabi009)
```
