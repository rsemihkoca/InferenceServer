import multiprocessing
import torch
from grpc_server import serve as grpc_serve
from config import load_config

config = load_config()

def run_grpc_server():
    grpc_serve()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    print("Checking CUDA availability" + str(torch.cuda.is_available()))
    if config['server']['http_port'] == config['server']['grpc_port']:
        raise ValueError("HTTP and GRPC ports must be different")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    print("Initializing server")
    grpc_process = multiprocessing.Process(target=run_grpc_server)
    print("Starting server")
    grpc_process.start()
    grpc_process.join()
    print("Server stopped")