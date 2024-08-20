import multiprocessing
from http_server import app as http_app
from grpc_server import serve as grpc_serve
from config import load_config
import uvicorn

config = load_config()


def run_http_server():
    uvicorn.run(http_app, host="0.0.0.0", port=config['server']['http_port'])


def run_grpc_server():
    grpc_serve()


if __name__ == "__main__":
    http_process = multiprocessing.Process(target=run_http_server)
    grpc_process = multiprocessing.Process(target=run_grpc_server)

    http_process.start()
    grpc_process.start()

    http_process.join()
    grpc_process.join()