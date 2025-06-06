from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from mpi4py import MPI
import cupy as cp
import os

class ParallelBackend(Enum):
    """Parallel computing backends"""
    CPU = "cpu"
    GPU = "gpu"
    MPI = "mpi"
    HYBRID = "hybrid"

@dataclass
class ParallelConfig:
    """Parallel computing configuration"""
    backend: ParallelBackend
    num_processes: int = 1
    num_threads: int = 1
    use_gpu: bool = False
    gpu_id: int = 0
    mpi_comm: Optional[MPI.Comm] = None
    
    def __post_init__(self):
        """Initialize configuration"""
        # Set number of processes
        if self.num_processes <= 0:
            self.num_processes = mp.cpu_count()
            
        # Set number of threads
        if self.num_threads <= 0:
            self.num_threads = mp.cpu_count() // self.num_processes
            
        # Initialize MPI if needed
        if self.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            if self.mpi_comm is None:
                self.mpi_comm = MPI.COMM_WORLD
                
        # Initialize GPU if needed
        if self.backend in [ParallelBackend.GPU, ParallelBackend.HYBRID]:
            if self.use_gpu:
                cp.cuda.Device(self.gpu_id).use()

class ParallelManager:
    def __init__(self, config: ParallelConfig):
        """
        Initialize parallel manager
        
        Args:
            config: Parallel configuration
        """
        self.config = config
        self.pool = None
        self.rank = 0
        self.size = 1
        
        # Initialize MPI
        if self.config.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            self.rank = self.config.mpi_comm.Get_rank()
            self.size = self.config.mpi_comm.Get_size()
            
        # Initialize process pool
        if self.config.backend in [ParallelBackend.CPU, ParallelBackend.HYBRID]:
            self.pool = mp.Pool(
                processes=self.config.num_processes,
                initializer=self._init_worker
            )
            
    def _init_worker(self):
        """Initialize worker process"""
        # Set number of threads
        os.environ["OMP_NUM_THREADS"] = str(self.config.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.num_threads)
        
    def map(self,
            func: Callable,
            data: List[Any],
            chunksize: Optional[int] = None) -> List[Any]:
        """
        Map function over data
        
        Args:
            func: Function to apply
            data: Input data
            chunksize: Optional chunk size
            
        Returns:
            Mapped results
        """
        if self.config.backend == ParallelBackend.CPU:
            return self.pool.map(func, data, chunksize=chunksize)
        elif self.config.backend == ParallelBackend.GPU:
            return self._gpu_map(func, data)
        elif self.config.backend == ParallelBackend.MPI:
            return self._mpi_map(func, data)
        elif self.config.backend == ParallelBackend.HYBRID:
            return self._hybrid_map(func, data)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
            
    def _gpu_map(self,
                 func: Callable,
                 data: List[Any]) -> List[Any]:
        """
        Map function over data using GPU
        
        Args:
            func: Function to apply
            data: Input data
            
        Returns:
            Mapped results
        """
        # Convert data to GPU arrays
        gpu_data = [cp.array(x) if isinstance(x, np.ndarray) else x for x in data]
        
        # Apply function
        results = [func(x) for x in gpu_data]
        
        # Convert results back to CPU
        return [cp.asnumpy(x) if isinstance(x, cp.ndarray) else x for x in results]
        
    def _mpi_map(self,
                 func: Callable,
                 data: List[Any]) -> List[Any]:
        """
        Map function over data using MPI
        
        Args:
            func: Function to apply
            data: Input data
            
        Returns:
            Mapped results
        """
        # Distribute data
        local_data = np.array_split(data, self.size)[self.rank]
        
        # Apply function
        local_results = [func(x) for x in local_data]
        
        # Gather results
        all_results = self.config.mpi_comm.allgather(local_results)
        
        # Flatten results
        return [x for sublist in all_results for x in sublist]
        
    def _hybrid_map(self,
                    func: Callable,
                    data: List[Any]) -> List[Any]:
        """
        Map function over data using hybrid approach
        
        Args:
            func: Function to apply
            data: Input data
            
        Returns:
            Mapped results
        """
        # Distribute data using MPI
        local_data = np.array_split(data, self.size)[self.rank]
        
        # Process local data using CPU pool
        local_results = self.pool.map(func, local_data)
        
        # Process results using GPU if available
        if self.config.use_gpu:
            local_results = self._gpu_map(func, local_results)
            
        # Gather results
        all_results = self.config.mpi_comm.allgather(local_results)
        
        # Flatten results
        return [x for sublist in all_results for x in sublist]
        
    def scatter(self,
                data: np.ndarray,
                root: int = 0) -> np.ndarray:
        """
        Scatter data
        
        Args:
            data: Input data
            root: Root process
            
        Returns:
            Local data
        """
        if self.config.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            # Get local size
            local_size = data.shape[0] // self.size
            if self.rank < data.shape[0] % self.size:
                local_size += 1
                
            # Create local array
            local_data = np.empty(local_size, dtype=data.dtype)
            
            # Scatter data
            self.config.mpi_comm.Scatterv(
                [data, self._get_counts(data.shape[0]), self._get_displs(data.shape[0])],
                local_data,
                root=root
            )
            
            return local_data
        else:
            return data
            
    def gather(self,
               data: np.ndarray,
               root: int = 0) -> Optional[np.ndarray]:
        """
        Gather data
        
        Args:
            data: Local data
            root: Root process
            
        Returns:
            Gathered data
        """
        if self.config.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            # Get total size
            total_size = self.config.mpi_comm.allreduce(data.shape[0])
            
            # Create output array on root
            if self.rank == root:
                gathered_data = np.empty(total_size, dtype=data.dtype)
            else:
                gathered_data = None
                
            # Gather data
            self.config.mpi_comm.Gatherv(
                data,
                [gathered_data, self._get_counts(total_size), self._get_displs(total_size)],
                root=root
            )
            
            return gathered_data
        else:
            return data
            
    def _get_counts(self, total_size: int) -> List[int]:
        """
        Get counts for scatter/gather
        
        Args:
            total_size: Total size
            
        Returns:
            List of counts
        """
        counts = [total_size // self.size] * self.size
        for i in range(total_size % self.size):
            counts[i] += 1
        return counts
        
    def _get_displs(self, total_size: int) -> List[int]:
        """
        Get displacements for scatter/gather
        
        Args:
            total_size: Total size
            
        Returns:
            List of displacements
        """
        counts = self._get_counts(total_size)
        displs = [0]
        for i in range(self.size - 1):
            displs.append(displs[-1] + counts[i])
        return displs
        
    def barrier(self):
        """Synchronize processes"""
        if self.config.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            self.config.mpi_comm.Barrier()
            
    def cleanup(self):
        """Cleanup parallel resources"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            
class ParallelArray:
    def __init__(self,
                 shape: Tuple[int, ...],
                 dtype: np.dtype = np.float64,
                 config: Optional[ParallelConfig] = None):
        """
        Initialize parallel array
        
        Args:
            shape: Array shape
            dtype: Data type
            config: Parallel configuration
        """
        self.shape = shape
        self.dtype = dtype
        self.config = config or ParallelConfig(ParallelBackend.CPU)
        self.manager = ParallelManager(self.config)
        
        # Create local array
        if self.config.backend in [ParallelBackend.MPI, ParallelBackend.HYBRID]:
            # Get local shape
            local_shape = list(shape)
            local_shape[0] = shape[0] // self.manager.size
            if self.manager.rank < shape[0] % self.manager.size:
                local_shape[0] += 1
                
            # Create local array
            self.local_array = np.empty(local_shape, dtype=dtype)
        else:
            self.local_array = np.empty(shape, dtype=dtype)
            
    def __getitem__(self, key):
        """Get array element"""
        return self.local_array[key]
        
    def __setitem__(self, key, value):
        """Set array element"""
        self.local_array[key] = value
        
    def gather(self, root: int = 0) -> Optional[np.ndarray]:
        """
        Gather array
        
        Args:
            root: Root process
            
        Returns:
            Gathered array
        """
        return self.manager.gather(self.local_array, root=root)
        
    def scatter(self, data: np.ndarray, root: int = 0):
        """
        Scatter array
        
        Args:
            data: Input data
            root: Root process
        """
        self.local_array = self.manager.scatter(data, root=root)
        
    def barrier(self):
        """Synchronize processes"""
        self.manager.barrier()
        
    def cleanup(self):
        """Cleanup resources"""
        self.manager.cleanup() 