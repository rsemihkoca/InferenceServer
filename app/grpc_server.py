import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
from core.model import YOLOv10X
from utils.config import load_config
from utils.logging import setup_logger
from utils.metrics import INFERENCE_TIME
import cv2
import numpy as np
import torch

config = load_config()
logger = setup_logger()
model = YOLOv10X(**config['model'])


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def Predict(self, request, context):
        nparr = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with INFERENCE_TIME.time():
            result = model.inference(img)

        logger.info("Prediction made for gRPC request")
        return inference_pb2.PredictionResponse(result=str(result))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"[::]:{config['server']['grpc_port']}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    torch.set_num_threads(1)  # Limit CPU threads as we're using GPU
    serve()