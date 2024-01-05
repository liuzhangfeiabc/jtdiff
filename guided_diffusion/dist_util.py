
import jittor as jt
from jittor import init
from jittor import nn
import io
import os
import socket
import blobfile as bf
import jittor.distributions as dist
from mpi4py import MPI
GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3

def setup_dist():
    return
    #xjh:dist
    if dist.is_initialized():
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{(MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE)}'
    comm = MPI.COMM_WORLD
    backend = ('gloo' if (not jt.cuda.is_available()) else 'nccl')
    if (backend == 'gloo'):
        hostname = 'localhost'
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ['MASTER_ADDR'] = comm.bcast(hostname, root=0)
    os.environ['RANK'] = str(comm.rank)
    os.environ['WORLD_SIZE'] = str(comm.size)
    port = comm.bcast(_find_free_port(), root=0)
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend=backend, init_method='env://')

def dev():
    if jt.cuda.is_available():
        return jt.device(f'cuda')
    return jt.device('cpu')

def load_state_dict(path, **kwargs):
    chunk_size = (2 ** 30)
    if (MPI.COMM_WORLD.Get_rank() == 0):
        with bf.BlobFile(path, 'rb') as f:
            data = f.read()
        num_chunks = (len(data) // chunk_size)
        if (len(data) % chunk_size):
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i:(i + chunk_size)])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)
    return jt.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    for p in params:
        with jt.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

