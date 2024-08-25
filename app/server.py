from PIL import Image
import numpy as np
import io

import grpc
from concurrent import futures
import time
import yaml
import torch
from ultralytics import YOLO
import inference_pb2
import inference_pb2_grpc
from utils.config_loader import load_config
from utils.logger import setup_logger
from utils.metrics import setup_metrics, update_inference_count, update_inference_latency

config = load_config('config.yml')
logger = setup_logger(config['logging']['level'], config['logging']['file'])

if config['metrics']['enabled']:
    setup_metrics(config['metrics']['port'])


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        logger.info("Loading model...")
        model = YOLO(config['model']['path'])
        model.to('cuda')  # Ensure model is on GPU
        return model

    def Predict(self, request, context):
        try:

            logger.info("Received an inference request")
            start_time = time.time()

            # Ensure CUDA is available
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. Please check your GPU setup.")


            image = Image.open(io.BytesIO(request.image_data))

            # Convert the PIL.Image to a numpy array
            image_np = np.array(image)

            # Process the input (assuming the request contains image data)
            results = self.model(image_np)

            # Process results and create response
            detections = []
            for r in results:
                for box in r.boxes:
                    if box.conf.item() > config['model']['confidence_threshold']:
                        detection = inference_pb2.Detection(
                            label=r.names[int(box.cls)],
                            confidence=box.conf.item(),
                            bbox=box.xyxy[0].tolist()
                        )
                        detections.append(detection)

            latency = time.time() - start_time
            logger.info(f"Inference completed in {latency:.4f} seconds")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return inference_pb2.PredictResponse(detections=detections)

        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during inference: {str(e)}")
            return inference_pb2.PredictResponse()


def serve():
    logger.info("Starting server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"{config['server']['host']}:{config['server']['port']}")
    server.start()
    logger.info(f"Server started on {config['server']['host']}:{config['server']['port']}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()