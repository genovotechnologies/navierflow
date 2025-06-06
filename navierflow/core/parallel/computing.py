import numpy as np
from mpi4py import MPI
import numba
from numba import prange
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

class ParallelMode(Enum):
    """Parallel computing modes"""
    MPI = "mpi"
    OPENMP = "openmp"
    HYBRID = "hybrid"

@dataclass
class ParallelConfig:
    """Parallel computing configuration"""
    mode: ParallelMode = ParallelMode.HYBRID
    num_threads: int = 4
    use_gpu: bool = True
    use_async: bool = True
    use_collective: bool = True
    use_nonblocking: bool = True
    use_persistent: bool = True

class ParallelManager:
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel computing manager
        
        Args:
            config: Parallel computing configuration
        """
        self.config = config or ParallelConfig()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._setup_parallel()
        
    def _setup_parallel(self):
        """Setup parallel computing environment"""
        if self.config.mode in [ParallelMode.OPENMP, ParallelMode.HYBRID]:
            numba.set_num_threads(self.config.num_threads)
            
    def scatter(self,
                data: np.ndarray,
                root: int = 0) -> np.ndarray:
        """
        Scatter data to all processes
        
        Args:
            data: Data to scatter
            root: Root process rank
            
        Returns:
            Local portion of data
        """
        if self.config.use_collective:
            return self.comm.Scatter(data, root=root)
        else:
            if self.rank == root:
                for i in range(self.size):
                    if i != root:
                        self.comm.Send(data[i], dest=i)
                return data[root]
            else:
                local_data = np.empty_like(data[0])
                self.comm.Recv(local_data, source=root)
                return local_data
                
    def gather(self,
               data: np.ndarray,
               root: int = 0) -> Optional[np.ndarray]:
        """
        Gather data from all processes
        
        Args:
            data: Local data to gather
            root: Root process rank
            
        Returns:
            Gathered data (only on root process)
        """
        if self.config.use_collective:
            return self.comm.Gather(data, root=root)
        else:
            if self.rank == root:
                gathered_data = np.empty((self.size,) + data.shape, dtype=data.dtype)
                gathered_data[root] = data
                for i in range(self.size):
                    if i != root:
                        self.comm.Recv(gathered_data[i], source=i)
                return gathered_data
            else:
                self.comm.Send(data, dest=root)
                return None
                
    def allreduce(self,
                 data: np.ndarray,
                 op: MPI.Op = MPI.SUM) -> np.ndarray:
        """
        Perform all-reduce operation
        
        Args:
            data: Local data
            op: Reduction operation
            
        Returns:
            Reduced data
        """
        if self.config.use_collective:
            return self.comm.Allreduce(data, op=op)
        else:
            reduced_data = np.empty_like(data)
            self.comm.Allreduce(data, reduced_data, op=op)
            return reduced_data
            
    def bcast(self,
              data: np.ndarray,
              root: int = 0) -> np.ndarray:
        """
        Broadcast data from root to all processes
        
        Args:
            data: Data to broadcast
            root: Root process rank
            
        Returns:
            Broadcasted data
        """
        if self.config.use_collective:
            return self.comm.Bcast(data, root=root)
        else:
            if self.rank == root:
                for i in range(self.size):
                    if i != root:
                        self.comm.Send(data, dest=i)
                return data
            else:
                self.comm.Recv(data, source=root)
                return data
                
    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def parallel_map(func: Callable,
                    data: np.ndarray,
                    *args) -> np.ndarray:
        """
        Apply function in parallel using OpenMP
        
        Args:
            func: Function to apply
            data: Input data
            *args: Additional arguments for func
            
        Returns:
            Result of parallel computation
        """
        result = np.empty_like(data)
        for i in prange(len(data)):
            result[i] = func(data[i], *args)
        return result
        
    def parallel_reduce(self,
                       data: np.ndarray,
                       op: Callable = np.sum) -> np.ndarray:
        """
        Perform parallel reduction
        
        Args:
            data: Input data
            op: Reduction operation
            
        Returns:
            Reduced result
        """
        # Local reduction
        local_result = op(data)
        
        # Global reduction
        if self.config.mode in [ParallelMode.MPI, ParallelMode.HYBRID]:
            global_result = self.allreduce(local_result)
        else:
            global_result = local_result
            
        return global_result
        
    def parallel_scan(self,
                     data: np.ndarray,
                     op: Callable = np.add) -> np.ndarray:
        """
        Perform parallel scan (prefix sum)
        
        Args:
            data: Input data
            op: Scan operation
            
        Returns:
            Scan result
        """
        # Local scan
        local_scan = np.empty_like(data)
        local_scan[0] = data[0]
        for i in range(1, len(data)):
            local_scan[i] = op(local_scan[i-1], data[i])
            
        # Global scan
        if self.config.mode in [ParallelMode.MPI, ParallelMode.HYBRID]:
            # Get last element of each process
            last_elements = np.array([local_scan[-1]])
            global_last = self.allreduce(last_elements)
            
            # Compute offset for each process
            offsets = np.zeros(self.size)
            for i in range(1, self.size):
                offsets[i] = offsets[i-1] + last_elements[i-1]
                
            # Apply offset
            local_scan += offsets[self.rank]
            
        return local_scan
        
    def parallel_sort(self,
                     data: np.ndarray,
                     axis: int = -1) -> np.ndarray:
        """
        Perform parallel sort
        
        Args:
            data: Input data
            axis: Axis to sort along
            
        Returns:
            Sorted data
        """
        # Local sort
        local_sorted = np.sort(data, axis=axis)
        
        # Global sort
        if self.config.mode in [ParallelMode.MPI, ParallelMode.HYBRID]:
            # Gather all data to root
            gathered_data = self.gather(local_sorted)
            
            if self.rank == 0:
                # Sort gathered data
                global_sorted = np.sort(gathered_data, axis=axis)
                
                # Scatter back to processes
                return self.scatter(global_sorted)
            else:
                return self.scatter(None)
        else:
            return local_sorted
            
    def parallel_filter(self,
                       data: np.ndarray,
                       condition: Callable) -> np.ndarray:
        """
        Perform parallel filter
        
        Args:
            data: Input data
            condition: Filter condition
            
        Returns:
            Filtered data
        """
        # Local filter
        local_filtered = data[condition(data)]
        
        # Global filter
        if self.config.mode in [ParallelMode.MPI, ParallelMode.HYBRID]:
            # Gather all filtered data to root
            gathered_data = self.gather(local_filtered)
            
            if self.rank == 0:
                # Concatenate filtered data
                global_filtered = np.concatenate(gathered_data)
                
                # Scatter back to processes
                return self.scatter(global_filtered)
            else:
                return self.scatter(None)
        else:
            return local_filtered 