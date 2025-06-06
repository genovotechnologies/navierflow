import numpy as np
import cupy as cp
import torch
from typing import Optional, Union, List, Tuple
from dataclasses import dataclass
from enum import Enum

class GPUMode(Enum):
    """GPU acceleration modes"""
    CUDA = "cuda"
    TENSOR = "tensor"
    MIXED = "mixed"

@dataclass
class GPUConfig:
    """GPU configuration parameters"""
    device_id: int = 0
    mode: GPUMode = GPUMode.MIXED
    use_tensor_cores: bool = True
    use_cudnn: bool = True
    memory_fraction: float = 0.9
    enable_async: bool = True
    enable_graphs: bool = True
    enable_multi_gpu: bool = False
    enable_amp: bool = True  # Automatic Mixed Precision

class GPUAccelerator:
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        Initialize GPU accelerator
        
        Args:
            config: GPU configuration parameters
        """
        self.config = config or GPUConfig()
        self._setup_device()
        self._setup_cudnn()
        self._setup_tensor_cores()
        
    def _setup_device(self):
        """Setup CUDA device and memory"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.device_id)
            torch.cuda.set_per_process_memory_fraction(
                self.config.memory_fraction,
                self.config.device_id
            )
            if self.config.enable_amp:
                torch.cuda.amp.autocast()
                
    def _setup_cudnn(self):
        """Setup cuDNN optimizations"""
        if self.config.use_cudnn:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
    def _setup_tensor_cores(self):
        """Setup Tensor Core optimizations"""
        if self.config.use_tensor_cores:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convert data to GPU tensor
        
        Args:
            data: Input data (numpy array or torch tensor)
            
        Returns:
            GPU tensor
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).cuda()
        elif isinstance(data, torch.Tensor):
            return data.cuda()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
            
    def to_cpu(self, data: torch.Tensor) -> np.ndarray:
        """
        Convert GPU tensor to CPU numpy array
        
        Args:
            data: GPU tensor
            
        Returns:
            CPU numpy array
        """
        return data.cpu().numpy()
        
    def compute_gradient(self,
                        field: torch.Tensor,
                        dx: float,
                        dy: float,
                        dz: Optional[float] = None) -> Tuple[torch.Tensor, ...]:
        """
        Compute gradient using GPU acceleration
        
        Args:
            field: Input field
            dx, dy, dz: Grid spacing
            
        Returns:
            Tuple of gradient components
        """
        if self.config.mode == GPUMode.TENSOR:
            return self._compute_gradient_tensor(field, dx, dy, dz)
        else:
            return self._compute_gradient_cuda(field, dx, dy, dz)
            
    def _compute_gradient_tensor(self,
                               field: torch.Tensor,
                               dx: float,
                               dy: float,
                               dz: Optional[float]) -> Tuple[torch.Tensor, ...]:
        """Compute gradient using Tensor Cores"""
        with torch.cuda.amp.autocast():
            if dz is None:  # 2D
                grad_x = torch.nn.functional.conv2d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_gradient_kernel(dx, 'x').cuda(),
                    padding=1
                ).squeeze()
                grad_y = torch.nn.functional.conv2d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_gradient_kernel(dy, 'y').cuda(),
                    padding=1
                ).squeeze()
                return grad_x, grad_y
            else:  # 3D
                grad_x = torch.nn.functional.conv3d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_gradient_kernel(dx, 'x').cuda(),
                    padding=1
                ).squeeze()
                grad_y = torch.nn.functional.conv3d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_gradient_kernel(dy, 'y').cuda(),
                    padding=1
                ).squeeze()
                grad_z = torch.nn.functional.conv3d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_gradient_kernel(dz, 'z').cuda(),
                    padding=1
                ).squeeze()
                return grad_x, grad_y, grad_z
                
    def _compute_gradient_cuda(self,
                             field: torch.Tensor,
                             dx: float,
                             dy: float,
                             dz: Optional[float]) -> Tuple[torch.Tensor, ...]:
        """Compute gradient using CUDA kernels"""
        if dz is None:  # 2D
            grad_x = cp.gradient(field.cpu().numpy(), dx, axis=1)
            grad_y = cp.gradient(field.cpu().numpy(), dy, axis=0)
            return (torch.from_numpy(cp.asnumpy(grad_x)).cuda(),
                   torch.from_numpy(cp.asnumpy(grad_y)).cuda())
        else:  # 3D
            grad_x = cp.gradient(field.cpu().numpy(), dx, axis=2)
            grad_y = cp.gradient(field.cpu().numpy(), dy, axis=1)
            grad_z = cp.gradient(field.cpu().numpy(), dz, axis=0)
            return (torch.from_numpy(cp.asnumpy(grad_x)).cuda(),
                   torch.from_numpy(cp.asnumpy(grad_y)).cuda(),
                   torch.from_numpy(cp.asnumpy(grad_z)).cuda())
                   
    def compute_laplacian(self,
                         field: torch.Tensor,
                         dx: float,
                         dy: float,
                         dz: Optional[float] = None) -> torch.Tensor:
        """
        Compute Laplacian using GPU acceleration
        
        Args:
            field: Input field
            dx, dy, dz: Grid spacing
            
        Returns:
            Laplacian of the field
        """
        if self.config.mode == GPUMode.TENSOR:
            return self._compute_laplacian_tensor(field, dx, dy, dz)
        else:
            return self._compute_laplacian_cuda(field, dx, dy, dz)
            
    def _compute_laplacian_tensor(self,
                                field: torch.Tensor,
                                dx: float,
                                dy: float,
                                dz: Optional[float]) -> torch.Tensor:
        """Compute Laplacian using Tensor Cores"""
        with torch.cuda.amp.autocast():
            if dz is None:  # 2D
                return torch.nn.functional.conv2d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_laplacian_kernel(dx, dy).cuda(),
                    padding=1
                ).squeeze()
            else:  # 3D
                return torch.nn.functional.conv3d(
                    field.unsqueeze(0).unsqueeze(0),
                    self._get_laplacian_kernel(dx, dy, dz).cuda(),
                    padding=1
                ).squeeze()
                
    def _compute_laplacian_cuda(self,
                              field: torch.Tensor,
                              dx: float,
                              dy: float,
                              dz: Optional[float]) -> torch.Tensor:
        """Compute Laplacian using CUDA kernels"""
        if dz is None:  # 2D
            laplacian = cp.gradient(cp.gradient(field.cpu().numpy(), dx, axis=1), dx, axis=1) + \
                       cp.gradient(cp.gradient(field.cpu().numpy(), dy, axis=0), dy, axis=0)
        else:  # 3D
            laplacian = cp.gradient(cp.gradient(field.cpu().numpy(), dx, axis=2), dx, axis=2) + \
                       cp.gradient(cp.gradient(field.cpu().numpy(), dy, axis=1), dy, axis=1) + \
                       cp.gradient(cp.gradient(field.cpu().numpy(), dz, axis=0), dz, axis=0)
        return torch.from_numpy(cp.asnumpy(laplacian)).cuda()
        
    def compute_fft(self,
                   field: torch.Tensor,
                   inverse: bool = False) -> torch.Tensor:
        """
        Compute FFT using GPU acceleration
        
        Args:
            field: Input field
            inverse: Whether to compute inverse FFT
            
        Returns:
            FFT of the field
        """
        if self.config.mode == GPUMode.TENSOR:
            return self._compute_fft_tensor(field, inverse)
        else:
            return self._compute_fft_cuda(field, inverse)
            
    def _compute_fft_tensor(self,
                          field: torch.Tensor,
                          inverse: bool) -> torch.Tensor:
        """Compute FFT using Tensor Cores"""
        with torch.cuda.amp.autocast():
            if inverse:
                return torch.fft.irfftn(field, dim=(-2, -1))
            else:
                return torch.fft.rfftn(field, dim=(-2, -1))
                
    def _compute_fft_cuda(self,
                         field: torch.Tensor,
                         inverse: bool) -> torch.Tensor:
        """Compute FFT using CUDA kernels"""
        if inverse:
            return torch.from_numpy(
                cp.asnumpy(cp.fft.irfftn(field.cpu().numpy()))
            ).cuda()
        else:
            return torch.from_numpy(
                cp.asnumpy(cp.fft.rfftn(field.cpu().numpy()))
            ).cuda()
            
    def _get_gradient_kernel(self,
                           dx: float,
                           axis: str) -> torch.Tensor:
        """Get gradient kernel for Tensor Core operations"""
        if axis == 'x':
            return torch.tensor([[-1, 0, 1]], dtype=torch.float32) / (2 * dx)
        elif axis == 'y':
            return torch.tensor([[-1], [0], [1]], dtype=torch.float32) / (2 * dx)
        else:  # z
            return torch.tensor([[[-1]], [[0]], [[1]]], dtype=torch.float32) / (2 * dx)
            
    def _get_laplacian_kernel(self,
                             dx: float,
                             dy: float,
                             dz: Optional[float] = None) -> torch.Tensor:
        """Get Laplacian kernel for Tensor Core operations"""
        if dz is None:  # 2D
            return torch.tensor([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ], dtype=torch.float32) / (dx * dy)
        else:  # 3D
            return torch.tensor([
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]],
                [[0, 1, 0],
                 [1, -6, 1],
                 [0, 1, 0]],
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ], dtype=torch.float32) / (dx * dy * dz) 