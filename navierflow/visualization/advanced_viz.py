import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from scipy import fft
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedVisualization:
    """
    Advanced visualization and post-processing tools:
    - Streamlines and pathlines
    - Vortex identification
    - Spectral analysis
    - Statistical analysis
    - Turbulence quantities
    - POD analysis
    - Interactive 3D visualization
    - Animation capabilities
    """
    def __init__(self, config: Dict = None):
        # Default configuration
        self.config = {
            'colormap': 'RdBu_r',
            'contour_levels': 50,
            'streamline_density': 2,
            'vector_skip': 5,
            'vortex_threshold': 0.1,
            'pod_modes': 10,
            'window_size': 1024,  # For spectral analysis
            'smoothing_factor': 0.5,
            'plot_style': 'dark_background'
        }
        if config:
            self.config.update(config)
            
        # Set plot style
        plt.style.use(self.config['plot_style'])
        
    def plot_flow_field(self, velocity: np.ndarray,
                       pressure: Optional[np.ndarray] = None,
                       vorticity: Optional[np.ndarray] = None,
                       streamlines: bool = True,
                       vectors: bool = True,
                       save_path: Optional[str] = None):
        """Plot complete flow field visualization"""
        fig = plt.figure(figsize=(15, 10))
        
        if pressure is not None and vorticity is not None:
            gs = plt.GridSpec(2, 2, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])
        else:
            gs = plt.GridSpec(1, 1, figure=fig)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = None
            ax3 = None
            
        # Velocity magnitude
        vel_mag = np.sqrt(velocity[..., 0]**2 + velocity[..., 1]**2)
        im1 = ax1.contourf(vel_mag, levels=self.config['contour_levels'],
                         cmap=self.config['colormap'])
        plt.colorbar(im1, ax=ax1, label='Velocity Magnitude')
        
        if streamlines:
            self._add_streamlines(ax1, velocity)
            
        if vectors:
            self._add_vectors(ax1, velocity)
            
        ax1.set_title('Velocity Field')
        
        if pressure is not None:
            im2 = ax2.contourf(pressure, levels=self.config['contour_levels'],
                            cmap='viridis')
            plt.colorbar(im2, ax=ax2, label='Pressure')
            ax2.set_title('Pressure Field')
            
        if vorticity is not None:
            im3 = ax3.contourf(vorticity, levels=self.config['contour_levels'],
                            cmap='RdBu_r')
            plt.colorbar(im3, ax=ax3, label='Vorticity')
            ax3.set_title('Vorticity Field')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_vortex_identification(self, velocity: np.ndarray,
                                 method: str = 'q_criterion',
                                 save_path: Optional[str] = None):
        """Plot vortex identification results"""
        if method == 'q_criterion':
            criterion = self._compute_q_criterion(velocity)
        elif method == 'lambda2':
            criterion = self._compute_lambda2(velocity)
        elif method == 'delta':
            criterion = self._compute_delta_criterion(velocity)
        else:
            raise ValueError(f"Unknown vortex identification method: {method}")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot criterion field
        im = ax.contourf(criterion, levels=self.config['contour_levels'],
                       cmap=self.config['colormap'])
        plt.colorbar(im, ax=ax, label=f'{method.replace("_", " ").title()}')
        
        # Add streamlines
        self._add_streamlines(ax, velocity)
        
        # Highlight vortex cores
        cores = criterion > self.config['vortex_threshold']
        ax.contour(cores, levels=[0.5], colors='white', linewidths=0.5)
        
        ax.set_title(f'Vortex Identification using {method.replace("_", " ").title()}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_spectral_analysis(self, velocity: np.ndarray,
                             dt: float,
                             point: Tuple[int, int],
                             save_path: Optional[str] = None):
        """Plot spectral analysis at a point"""
        # Extract time series at point
        u = velocity[:, point[0], point[1], 0]
        v = velocity[:, point[0], point[1], 1]
        
        # Compute FFT
        window = np.hanning(len(u))
        fu = fft.fft(u * window)
        fv = fft.fft(v * window)
        
        # Frequency axis
        freq = fft.fftfreq(len(u), dt)
        
        # Power spectral density
        psd_u = np.abs(fu)**2
        psd_v = np.abs(fv)**2
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Time series
        ax1.plot(np.arange(len(u))*dt, u, label='u')
        ax1.plot(np.arange(len(v))*dt, v, label='v')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Velocity')
        ax1.legend()
        ax1.set_title('Velocity Time Series')
        
        # Power spectrum
        ax2.semilogy(freq[freq >= 0], psd_u[freq >= 0], label='u')
        ax2.semilogy(freq[freq >= 0], psd_v[freq >= 0], label='v')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('PSD')
        ax2.legend()
        ax2.set_title('Power Spectral Density')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_pod_analysis(self, velocity: np.ndarray,
                         n_modes: Optional[int] = None,
                         save_path: Optional[str] = None):
        """Plot Proper Orthogonal Decomposition analysis"""
        if n_modes is None:
            n_modes = self.config['pod_modes']
            
        # Reshape velocity field for POD
        n_snapshots = len(velocity)
        n_points = velocity.shape[1] * velocity.shape[2]
        X = velocity.reshape(n_snapshots, -1)
        
        # Compute POD
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        
        # Energy content
        energy = S**2 / np.sum(S**2)
        cumulative_energy = np.cumsum(energy)
        
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # Energy content
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(energy[:n_modes] * 100, 'o-')
        ax1.set_xlabel('Mode Number')
        ax1.set_ylabel('Energy Content (%)')
        ax1.set_title('POD Mode Energy Content')
        ax1.grid(True)
        
        # Cumulative energy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(cumulative_energy[:n_modes] * 100, 'o-')
        ax2.set_xlabel('Mode Number')
        ax2.set_ylabel('Cumulative Energy (%)')
        ax2.set_title('POD Cumulative Energy')
        ax2.grid(True)
        
        # Mode shapes
        ax3 = fig.add_subplot(gs[1, :])
        mode_shape = Vh[0].reshape(velocity.shape[1:])
        magnitude = np.sqrt(mode_shape[..., 0]**2 + mode_shape[..., 1]**2)
        im = ax3.contourf(magnitude, levels=self.config['contour_levels'],
                        cmap=self.config['colormap'])
        plt.colorbar(im, ax=ax3)
        self._add_streamlines(ax3, mode_shape)
        ax3.set_title('First POD Mode')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, (U, S, Vh)
        
    def plot_turbulence_statistics(self, velocity: np.ndarray,
                                 save_path: Optional[str] = None):
        """Plot turbulence statistics"""
        # Compute statistics
        mean_u = np.mean(velocity[..., 0], axis=0)
        mean_v = np.mean(velocity[..., 1], axis=0)
        
        fluctuations_u = velocity[..., 0] - mean_u
        fluctuations_v = velocity[..., 1] - mean_v
        
        rms_u = np.sqrt(np.mean(fluctuations_u**2, axis=0))
        rms_v = np.sqrt(np.mean(fluctuations_v**2, axis=0))
        
        reynolds_stress = np.mean(fluctuations_u * fluctuations_v, axis=0)
        
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # Mean velocity
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(np.sqrt(mean_u**2 + mean_v**2),
                         levels=self.config['contour_levels'],
                         cmap=self.config['colormap'])
        plt.colorbar(im1, ax=ax1, label='Mean Velocity Magnitude')
        self._add_streamlines(ax1, np.stack([mean_u, mean_v], axis=-1))
        ax1.set_title('Mean Velocity Field')
        
        # RMS fluctuations
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(np.sqrt(rms_u**2 + rms_v**2),
                         levels=self.config['contour_levels'],
                         cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='RMS Fluctuations')
        ax2.set_title('RMS Velocity Fluctuations')
        
        # Reynolds stress
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.contourf(reynolds_stress,
                         levels=self.config['contour_levels'],
                         cmap='RdBu_r')
        plt.colorbar(im3, ax=ax3, label='Reynolds Stress')
        ax3.set_title('Reynolds Stress')
        
        # Joint PDF of velocity fluctuations
        ax4 = fig.add_subplot(gs[1, 1])
        xy = np.vstack([fluctuations_u.flatten(), fluctuations_v.flatten()])
        z = gaussian_kde(xy)(xy)
        
        scatter = ax4.scatter(fluctuations_u.flatten(),
                           fluctuations_v.flatten(),
                           c=z, s=1, cmap='viridis')
        plt.colorbar(scatter, ax=ax4, label='Density')
        ax4.set_xlabel("u'")
        ax4.set_ylabel("v'")
        ax4.set_title('Joint PDF of Velocity Fluctuations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_visualization(self, velocity: np.ndarray,
                                      pressure: Optional[np.ndarray] = None,
                                      vorticity: Optional[np.ndarray] = None):
        """Create interactive visualization using plotly"""
        vel_mag = np.sqrt(velocity[..., 0]**2 + velocity[..., 1]**2)
        
        if pressure is not None and vorticity is not None:
            fig = make_subplots(rows=2, cols=2,
                              subplot_titles=('Velocity Magnitude',
                                           'Pressure',
                                           'Vorticity'))
        else:
            fig = go.Figure()
            
        # Velocity magnitude
        fig.add_trace(
            go.Heatmap(z=vel_mag,
                      colorscale='RdBu_r',
                      name='Velocity Magnitude'),
            row=1, col=1
        )
        
        if pressure is not None:
            fig.add_trace(
                go.Heatmap(z=pressure,
                          colorscale='Viridis',
                          name='Pressure'),
                row=1, col=2
            )
            
        if vorticity is not None:
            fig.add_trace(
                go.Heatmap(z=vorticity,
                          colorscale='RdBu',
                          name='Vorticity'),
                row=2, col=1
            )
            
        fig.update_layout(
            title='Flow Field Visualization',
            height=800,
            showlegend=True
        )
        
        return fig
        
    def _add_streamlines(self, ax: plt.Axes,
                        velocity: np.ndarray):
        """Add streamlines to plot"""
        Y, X = np.mgrid[0:velocity.shape[0], 0:velocity.shape[1]]
        ax.streamplot(X, Y,
                     velocity[..., 0], velocity[..., 1],
                     density=self.config['streamline_density'],
                     color='white',
                     linewidth=0.5)
                     
    def _add_vectors(self, ax: plt.Axes,
                    velocity: np.ndarray):
        """Add velocity vectors to plot"""
        skip = self.config['vector_skip']
        Y, X = np.mgrid[0:velocity.shape[0], 0:velocity.shape[1]]
        ax.quiver(X[::skip, ::skip],
                 Y[::skip, ::skip],
                 velocity[::skip, ::skip, 0],
                 velocity[::skip, ::skip, 1],
                 scale=50, color='white', alpha=0.5)
                 
    def _compute_q_criterion(self, velocity: np.ndarray) -> np.ndarray:
        """Compute Q-criterion"""
        dudx = np.gradient(velocity[..., 0], axis=1)
        dudy = np.gradient(velocity[..., 0], axis=0)
        dvdx = np.gradient(velocity[..., 1], axis=1)
        dvdy = np.gradient(velocity[..., 1], axis=0)
        
        # Compute rate-of-strain and vorticity tensors
        S11 = dudx
        S12 = 0.5 * (dudy + dvdx)
        S21 = S12
        S22 = dvdy
        
        O11 = 0
        O12 = 0.5 * (dudy - dvdx)
        O21 = -O12
        O22 = 0
        
        # Compute Q-criterion
        Q = -0.5 * (
            (S11**2 + S12**2 + S21**2 + S22**2) -
            (O11**2 + O12**2 + O21**2 + O22**2)
        )
        
        return Q
        
    def _compute_lambda2(self, velocity: np.ndarray) -> np.ndarray:
        """Compute lambda2 criterion"""
        dudx = np.gradient(velocity[..., 0], axis=1)
        dudy = np.gradient(velocity[..., 0], axis=0)
        dvdx = np.gradient(velocity[..., 1], axis=1)
        dvdy = np.gradient(velocity[..., 1], axis=0)
        
        # Compute S^2 + Ω^2
        S11 = dudx
        S12 = 0.5 * (dudy + dvdx)
        S21 = S12
        S22 = dvdy
        
        O11 = 0
        O12 = 0.5 * (dudy - dvdx)
        O21 = -O12
        O22 = 0
        
        # Compute eigenvalues
        a = 1
        b = -(S11**2 + 2*S12*S21 + S22**2 +
             O11**2 + 2*O12*O21 + O22**2)
        c = (S11*S22 - S12*S21)**2 + \
            (O11*O22 - O12*O21)**2 + \
            2*(S11*O22 + S22*O11 - S12*O21 - S21*O12)
            
        lambda2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        
        return lambda2
        
    def _compute_delta_criterion(self, velocity: np.ndarray) -> np.ndarray:
        """Compute delta criterion"""
        dudx = np.gradient(velocity[..., 0], axis=1)
        dudy = np.gradient(velocity[..., 0], axis=0)
        dvdx = np.gradient(velocity[..., 1], axis=1)
        dvdy = np.gradient(velocity[..., 1], axis=0)
        
        # Compute velocity gradient tensor invariants
        P = -(dudx + dvdy)
        Q = dudx*dvdy - dudy*dvdx
        R = -dudx*dvdy*dvdy + dudy*dvdx*dvdy
        
        # Compute discriminant
        delta = (27/4) * R**2 + P**2 * Q**2 - (4/27) * P**3 * R - 4 * Q**3
        
        return delta 