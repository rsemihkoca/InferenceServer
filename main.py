import io
import time
from concurrent import futures

import grpc
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

import proto.inference_pb2 as inference_pb2
import proto.inference_pb2_grpc as inference_pb2_grpc
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
        model.to('cuda')
        return model

    def process_result(self, result):
        detections = []
        for box in result.boxes:
            if box.conf.item() > config['model']['confidence_threshold']:
                detection = inference_pb2.Detection(
                    label=result.names[int(box.cls)],
                    confidence=box.conf.item(),
                    bbox=box.xyxy[0].tolist()
                )
                detections.append(detection)
        return detections

    def Predict(self, request, context):
        try:
            logger.info("Received a single inference request")
            start_time = time.time()

            image = Image.open(io.BytesIO(request.image_data))
            image_np = np.array(image)

            results = self.model(image_np)
            detections = self.process_result(results[0])

            response = inference_pb2.PredictResponse(detections=detections)

            latency = time.time() - start_time
            logger.info(f"Single inference completed in {latency:.4f} seconds")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return response

        except Exception as e:
            logger.error(f"Error during single inference: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during inference: {str(e)}")
            return inference_pb2.PredictResponse()

    def BatchPredict(self, request, context):
        try:
            logger.info(f"Received a batch inference request with {len(request.image_data)} images")
            start_time = time.time()

            images = [np.array(Image.open(io.BytesIO(img))) for img in request.image_data]
            results = self.model(images, batch=len(images))

            batch_response = inference_pb2.BatchPredictResponse()
            for result in results:
                detections = self.process_result(result)
                batch_response.results.append(inference_pb2.PredictResponse(detections=detections))

            latency = time.time() - start_time
            logger.info(f"Batch inference completed in {latency:.4f} seconds")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return batch_response

        except Exception as e:
            logger.error(f"Error during batch inference: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during batch inference: {str(e)}")
            return inference_pb2.BatchPredictResponse()

    def TestPredict(self, request, context):
        try:
            logger.info("Received a test inference request")
            start_time = time.time()

            image = Image.open(io.BytesIO(request.image_data))
            image_np = np.array(image)

            results = self.model(image_np)
            result = results[0]

            detections = self.process_result(result)

            plot_image = result.plot()
            plot_image_bytes = io.BytesIO()
            Image.fromarray(plot_image).save(plot_image_bytes, format='PNG')

            response = inference_pb2.TestPredictResponse(
                detections=detections,
                orig_shape=str(result.orig_shape),
                boxes=str(result.boxes),
                probs=str(result.probs),
                keypoints=str(result.keypoints),
                obb=str(result.obb),
                speed=str(result.speed),
                names=str(result.names),
                json_output=result.tojson(),
                plot_image=plot_image_bytes.getvalue(),
                verbose_output=result.verbose()
            )

            latency = time.time() - start_time
            logger.info(f"Test inference completed in {latency:.4f} seconds")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return response

        except Exception as e:
            logger.error(f"Error during test inference: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during test inference: {str(e)}")
            return inference_pb2.TestPredictResponse()

def serve():
    logger.info("Starting server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"{config['server']['host']}:{config['server']['port']}")
    server.start()
    logger.info(f"Server started on {config['server']['host']}:{config['server']['port']} version {config['server']['version']}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()