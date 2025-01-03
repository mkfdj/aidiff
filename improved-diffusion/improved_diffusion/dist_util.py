"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist
import torch_xla.core.xla_model as xm

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1  # Adjust as needed for your setup

SETUP_RETRY_COUNT = 3

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    
    # Check for available XLA devices
    devices = xm.get_xla_supported_devices()
    backend = "xla" if devices else "gloo"

    # Determine hostname based on backend
    hostname = socket.gethostbyname(socket.getfqdn()) if backend == "xla" else "localhost"
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    # Find and broadcast a free port for communication
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)

    # Initialize the process group based on the selected backend
    if backend == "xla":
        print("Using TPU with XLA backend.")
        # Additional TPU configuration can be added here if needed
        # Example: xm.init() or other TPU-specific settings can be added here.
        
    else:
        print("Using Gloo backend for CPU/GPU.")
        dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    return xm.xla_device()

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    """Find a free port on the system."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
