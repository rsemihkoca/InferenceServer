import grpc
from concurrent import futures
import torch
from inference_pb2 import PredictionResponse
from inference_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
from model import YOLOv10X
from config import load_config
from logger import setup_logger
from metrics import INFERENCE_TIME
import cv2
import numpy as np

config = load_config()
logger = setup_logger()
model = YOLOv10X(**config['model'])

class InferenceService(InferenceServiceServicer):
    def Predict(self, request, context):
        try:
            print("Received request" + str(request))
            nparr = np.frombuffer(request.image, np.uint8)
            if nparr.size == 0:
                raise ValueError("Empty image data received")
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")
            print(f"Decoded image shape: {img.shape}")
            with INFERENCE_TIME.time():
                result = model.inference(img)
            logger.info("Prediction made for gRPC request")
            return PredictionResponse(result=str(result))
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return PredictionResponse(result=f"Error: {str(e)}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"[::]:{config['server']['grpc_port']}")
    server.start()
    print(f"Server started at port {config['server']['grpc_port']}")
    server.wait_for_termination()