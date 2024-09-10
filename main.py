import io
import time
from concurrent import futures

import grpc
import numpy as np
from PIL import Image
from ultralytics import YOLOv10
import torch
import cv2

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
        self.target_classes = self._get_target_classes()
        self.class_names = config['model']['classNames']
        self.confidence_threshold = config['model']['confidence_threshold']
        # self.nms_threshold = config['model']['nms_threshold']
        self.model = self._load_model()

    def _load_model(self):
        logger.info("Loading model...")
        try:
            model = YOLOv10(config['model']['path'], task='detect')
            model.to(torch.device('cuda'))
            # model.iou = self.nms_threshold
            model.agnostic = True  # NMS for all classes
            model.multi_label = True  # Single class per box
            model.max_det = 300  # Maximum number of detections
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _get_target_classes(self):
        return [int(class_id) for class_id in config['model']['classNames'].keys()]

    @torch.no_grad()
    def _process_image(self, image_data, filter_classes=True):
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        classes = self.target_classes if filter_classes else None
        results = self.model(image_np, 
                             conf=self.confidence_threshold, 
                             classes=classes)
        return results[0]

    def _process_result(self, result):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, scores, class_ids

    def _create_detections(self, boxes, scores, class_ids):
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            xyxy = box.tolist()
            centroid = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            detection = inference_pb2.Detection(
                label=self.class_names.get(str(class_id), "Unknown"),
                confidence=float(score),
                bbox=xyxy,
                centroid=centroid
            )
            detections.append(detection)
        return detections

    def _create_plot_image(self, result, detections):
        # Get the plotted image from the YOLOv10 result
        plot_image = result.plot(boxes=True, conf=True, line_width=2)
        
        # Add centroids to the plot
        for detection in detections:
            centroid = detection.centroid
            cv2.circle(plot_image, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)
        
        return plot_image
    
    def Predict(self, request, context):
        camera_ip = request.image.camera_ip
        logger.info(f"Received a single inference request from camera IP: {camera_ip}")
        start_time = time.time()

        try:
            result = self._process_image(request.image.image_data, filter_classes=True)
            boxes, scores, class_ids = self._process_result(result)
            detections = self._create_detections(boxes, scores, class_ids)

            response = inference_pb2.PredictResponse(
                camera_ip=camera_ip, 
                detections=detections
            )

            self._log_and_update_metrics(time.time() - start_time, camera_ip)
            return response

        except Exception as e:
            self._handle_error(e, context, f"Error during single inference for camera IP {camera_ip}")
            return inference_pb2.PredictResponse()

    def PredictX(self, request, context):
        camera_ip = request.image.camera_ip
        logger.info(f"Received a PredictX request from camera IP: {camera_ip}")
        start_time = time.time()

        try:
            result = self._process_image(request.image.image_data, filter_classes=True)
            boxes, scores, class_ids = self._process_result(result)
            
            names = [self.class_names.get(str(class_id), "Unknown") for class_id in class_ids]
            centroids = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes]

            response = inference_pb2.PredictXResponse(
                camera_ip=camera_ip,
                boxes=boxes.flatten().tolist(),
                classes=class_ids.tolist(),
                confidences=scores.tolist(),
                names=names,
                centroids=np.array(centroids).flatten().tolist()
            )

            self._log_and_update_metrics(time.time() - start_time, camera_ip, "PredictX")
            return response

        except Exception as e:
            self._handle_error(e, context, f"Error during PredictX for camera IP {camera_ip}")
            return inference_pb2.PredictXResponse()

    def BatchPredict(self, request, context):
        logger.info(f"Received a batch inference request with {len(request.images)} images")
        start_time = time.time()

        try:
            batch_response = inference_pb2.BatchPredictResponse()
            batch_images = [np.array(Image.open(io.BytesIO(img.image_data))) for img in request.images]
            
            with torch.no_grad():
                batch_results = self.model(batch_images, classes=self.target_classes, conf=self.confidence_threshold)

            for img_data, result in zip(request.images, batch_results):
                boxes, scores, class_ids = self._process_result(result)
                detections = self._create_detections(boxes, scores, class_ids)
                batch_response.results.append(inference_pb2.PredictResponse(camera_ip=img_data.camera_ip, detections=detections))

            self._log_and_update_metrics(time.time() - start_time, "Batch")
            return batch_response

        except Exception as e:
            self._handle_error(e, context, "Error during batch inference")
            return inference_pb2.BatchPredictResponse()

    def TestPredict(self, request, context):
        camera_ip = request.image.camera_ip
        logger.info(f"Received a test inference request from camera IP: {camera_ip}")
        start_time = time.time()

        try:
            result = self._process_image(request.image.image_data, filter_classes=False)
            boxes, scores, class_ids = self._process_result(result)
            detections = self._create_detections(boxes, scores, class_ids)
            
            # Create the plot image with bounding boxes and centroids
            plot_image = self._create_plot_image(result, detections)

            # Convert plot image to bytes
            plot_image_bytes = io.BytesIO()
            Image.fromarray(plot_image).save(plot_image_bytes, format='PNG')


            response = inference_pb2.TestPredictResponse(
                camera_ip=camera_ip,
                detections=detections,
                orig_shape=str(result.orig_shape),
                boxes=str(result.boxes),
                speed=str(result.speed),
                names=str(result.names),
                plot_image=plot_image_bytes.getvalue(),
                json_output=result.tojson(),
                verbose_output=result.verbose()
            )

            self._log_and_update_metrics(time.time() - start_time, camera_ip, "Test")
            return response

        except Exception as e:
            self._handle_error(e, context, f"Error during test inference for camera IP {camera_ip}")
            return inference_pb2.TestPredictResponse()

    def _log_and_update_metrics(self, latency, camera_ip, prefix=""):
        logger.info(f"{prefix} inference completed in {latency:.4f} seconds for camera IP: {camera_ip}")
        if config['metrics']['enabled']:
            update_inference_count()
            update_inference_latency(latency)

    def _handle_error(self, error, context, message):
        logger.error(f"{message}: {str(error)}")
        context.set_code(grpc.StatusCode.INTERNAL)
        context.set_details(f"An error occurred: {str(error)}")

def serve():
    logger.info("Starting server...")
    try:
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ('grpc.max_receive_message_length', 40 * 1024 * 1024),
                ('grpc.max_send_message_length', 40 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 5000),
            ]
        )
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
        server.add_insecure_port(f"{config['server']['host']}:{config['server']['port']}")
        server.start()
        logger.info(f"Server started on {config['server']['host']}:{config['server']['port']} version {config['server']['version']}")
        server.wait_for_termination()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        serve()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        import sys
        sys.exit(1)