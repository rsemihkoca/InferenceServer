import multiprocessing

import torch
print(torch.cuda.is_available())
#from http_server import app as http_app
from grpc_server import serve as grpc_serve
from config import load_config
#import uvicorn

config = load_config()


def run_http_server():
    uvicorn.run(http_app, host="0.0.0.0", port=config['server']['http_port'])


def run_grpc_server():
    grpc_serve()


if __name__ == "__main__":
    if config['server']['http_port'] == config['server']['grpc_port']:
        raise ValueError("HTTP and GRPC ports must be different")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    print("Initializing server")
#    http_process = multiprocessing.Process(target=run_http_server)
    grpc_process = multiprocessing.Process(target=run_grpc_server)
    print("Starting server")
#    http_process.start()
    grpc_process.start()
    print("Server started on ports: HTTP: {}, GRPC: {}".format(config['server']['http_port'], config['server']['grpc_port']))
#    http_process.join()
    grpc_process.join()
    print("Server stopped")
