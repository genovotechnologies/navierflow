import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Optional, Tuple
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

def create_velocity_plot(
    velocity: np.ndarray,
    colormap: str = "viridis",
    show_streamlines: bool = True,
    show_vectors: bool = False,
    smoothing: float = 0.5,
    vector_density: int = 20
) -> go.Figure:
    """Create velocity field visualization"""
    # Compute velocity magnitude
    velocity_mag = np.sqrt(velocity[:, :, 0]**2 + velocity[:, :, 1]**2)
    
    # Apply smoothing
    if smoothing > 0:
        velocity_mag = gaussian_filter(velocity_mag, sigma=smoothing)
    
    # Create base heatmap
    fig = go.Figure()
    
    # Add velocity magnitude heatmap
    fig.add_trace(go.Heatmap(
        z=velocity_mag,
        colorscale=colormap,
        showscale=True,
        colorbar=dict(
            title="Velocity Magnitude (m/s)",
            titleside="right"
        )
    ))
    
    if show_streamlines:
        # Compute streamlines
        y, x = np.mgrid[0:velocity.shape[0], 0:velocity.shape[1]]
        streamline_density = 2
        
        fig.add_trace(go.Streamtube(
            x=x[::streamline_density, ::streamline_density],
            y=y[::streamline_density, ::streamline_density],
            u=velocity[::streamline_density, ::streamline_density, 0],
            v=velocity[::streamline_density, ::streamline_density, 1],
            w=np.zeros_like(velocity[::streamline_density, ::streamline_density, 0]),
            colorscale=colormap,
            showscale=False,
            maxdisplayed=1000
        ))
    
    if show_vectors:
        # Add velocity vectors
        y, x = np.mgrid[0:velocity.shape[0]:vector_density, 0:velocity.shape[1]:vector_density]
        u = velocity[::vector_density, ::vector_density, 0]
        v = velocity[::vector_density, ::vector_density, 1]
        
        # Normalize vectors
        magnitude = np.sqrt(u**2 + v**2)
        u = u / (magnitude + 1e-10)
        v = v / (magnitude + 1e-10)
        
        fig.add_trace(go.Quiver(
            x=x,
            y=y,
            u=u,
            v=v,
            scale=0.1,
            line=dict(width=1),
            name="Velocity Vectors"
        ))
    
    # Update layout
    fig.update_layout(
        title="Velocity Field",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_pressure_plot(
    pressure: np.ndarray,
    colormap: str = "viridis",
    smoothing: float = 0.5,
    show_contours: bool = True
) -> go.Figure:
    """Create pressure field visualization"""
    # Apply smoothing
    if smoothing > 0:
        pressure = gaussian_filter(pressure, sigma=smoothing)
    
    fig = go.Figure()
    
    # Add pressure heatmap
    fig.add_trace(go.Heatmap(
        z=pressure,
        colorscale=colormap,
        showscale=True,
        colorbar=dict(
            title="Pressure (Pa)",
            titleside="right"
        )
    ))
    
    if show_contours:
        # Add contour lines
        fig.add_trace(go.Contour(
            z=pressure,
            colorscale=colormap,
            showscale=False,
            contours=dict(
                start=np.min(pressure),
                end=np.max(pressure),
                size=(np.max(pressure) - np.min(pressure)) / 20
            ),
            line=dict(width=0.5),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Pressure Field",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_temperature_plot(
    temperature: np.ndarray,
    colormap: str = "inferno",
    smoothing: float = 0.5,
    show_isotherms: bool = True
) -> go.Figure:
    """Create temperature field visualization"""
    # Apply smoothing
    if smoothing > 0:
        temperature = gaussian_filter(temperature, sigma=smoothing)
    
    fig = go.Figure()
    
    # Add temperature heatmap
    fig.add_trace(go.Heatmap(
        z=temperature,
        colorscale=colormap,
        showscale=True,
        colorbar=dict(
            title="Temperature (K)",
            titleside="right"
        )
    ))
    
    if show_isotherms:
        # Add isotherms
        fig.add_trace(go.Contour(
            z=temperature,
            colorscale=colormap,
            showscale=False,
            contours=dict(
                start=np.min(temperature),
                end=np.max(temperature),
                size=(np.max(temperature) - np.min(temperature)) / 15
            ),
            line=dict(width=0.5),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Temperature Field",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_vorticity_plot(
    vorticity: np.ndarray,
    colormap: str = "RdBu",
    smoothing: float = 0.5,
    show_contours: bool = True
) -> go.Figure:
    """Create vorticity field visualization"""
    # Apply smoothing
    if smoothing > 0:
        vorticity = gaussian_filter(vorticity, sigma=smoothing)
    
    fig = go.Figure()
    
    # Symmetric colorscale around zero
    max_abs = np.max(np.abs(vorticity))
    
    # Add vorticity heatmap
    fig.add_trace(go.Heatmap(
        z=vorticity,
        colorscale=colormap,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        showscale=True,
        colorbar=dict(
            title="Vorticity (1/s)",
            titleside="right"
        )
    ))
    
    if show_contours:
        # Add contour lines
        fig.add_trace(go.Contour(
            z=vorticity,
            colorscale=colormap,
            showscale=False,
            contours=dict(
                start=-max_abs,
                end=max_abs,
                size=max_abs/10
            ),
            line=dict(width=0.5),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="Vorticity Field",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_energy_spectrum_plot(
    velocity: np.ndarray,
    dx: float = 1.0
) -> go.Figure:
    """Create energy spectrum plot"""
    # Compute 2D FFT of velocity field
    u_fft = np.fft.fft2(velocity[:, :, 0])
    v_fft = np.fft.fft2(velocity[:, :, 1])
    
    # Compute wavenumbers
    kx = 2 * np.pi * np.fft.fftfreq(velocity.shape[0], dx)
    ky = 2 * np.pi * np.fft.fftfreq(velocity.shape[1], dx)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_mesh**2 + ky_mesh**2)
    
    # Compute energy spectrum
    energy_density = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)
    
    # Bin the energy density by wavenumber magnitude
    k_bins = np.linspace(0, np.max(k_mag), 50)
    energy_spectrum = np.zeros_like(k_bins[:-1])
    
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        energy_spectrum[i] = np.mean(energy_density[mask])
    
    # Create plot
    fig = go.Figure()
    
    # Add energy spectrum
    fig.add_trace(go.Scatter(
        x=k_bins[:-1],
        y=energy_spectrum,
        mode='lines',
        name='Energy Spectrum'
    ))
    
    # Add k^(-5/3) line for comparison
    k_range = np.logspace(np.log10(k_bins[1]), np.log10(k_bins[-2]), 100)
    kolmogorov = k_range**(-5/3) * energy_spectrum[0] / (k_bins[1]**(-5/3))
    
    fig.add_trace(go.Scatter(
        x=k_range,
        y=kolmogorov,
        mode='lines',
        name='k^(-5/3)',
        line=dict(dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="Energy Spectrum",
        xaxis_title="Wavenumber (k)",
        yaxis_title="Energy Density",
        xaxis_type="log",
        yaxis_type="log",
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_q_criterion_plot(
    velocity: np.ndarray,
    dx: float = 1.0,
    threshold: float = 0.1
) -> go.Figure:
    """Create Q-criterion visualization for vortex identification"""
    # Compute velocity gradients
    dudx = np.gradient(velocity[:, :, 0], dx, axis=1)
    dudy = np.gradient(velocity[:, :, 0], dx, axis=0)
    dvdx = np.gradient(velocity[:, :, 1], dx, axis=1)
    dvdy = np.gradient(velocity[:, :, 1], dx, axis=0)
    
    # Compute strain rate tensor
    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S21 = S12
    S22 = dvdy
    
    # Compute rotation tensor
    R11 = 0
    R12 = 0.5 * (dudy - dvdx)
    R21 = -R12
    R22 = 0
    
    # Compute Q-criterion
    Q = -0.5 * (
        np.sum(S11**2 + S12**2 + S21**2 + S22**2) -
        np.sum(R11**2 + R12**2 + R21**2 + R22**2)
    )
    
    # Create plot
    fig = go.Figure()
    
    # Add Q-criterion isosurfaces
    fig.add_trace(go.Isosurface(
        x=np.arange(velocity.shape[0]),
        y=np.arange(velocity.shape[1]),
        z=np.zeros_like(Q),
        value=Q,
        isomin=threshold,
        isomax=np.max(Q),
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(
            title="Q-criterion",
            titleside="right"
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Q-criterion Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig 