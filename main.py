import io
import time
from concurrent import futures

import grpc
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import torch

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
        self.target_classes = self.get_target_classes()
        self.class_names = config['model']['classNames']

    def load_model(self):
        logger.info("Loading model...")
        model = YOLO(config['model']['path'])
        model.to('cuda')
        return model

    def get_target_classes(self):
        return [int(class_id) for class_id in config['model']['classNames'].keys()]

    def custom_nms(self, boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def custom_postprocess(self, result):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        nms_threshold = config['model']['nms_threshold']
        confidence_threshold = config['model']['confidence_threshold']

        final_boxes = []
        final_scores = []
        final_class_ids = []

        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            keep = self.custom_nms(class_boxes, class_scores, nms_threshold)

            final_boxes.extend(class_boxes[keep])
            final_scores.extend(class_scores[keep])
            final_class_ids.extend([class_id] * len(keep))

        final_boxes = np.array(final_boxes)
        final_scores = np.array(final_scores)
        final_class_ids = np.array(final_class_ids)

        mask = final_scores > confidence_threshold
        final_boxes = final_boxes[mask]
        final_scores = final_scores[mask]
        final_class_ids = final_class_ids[mask]

        return final_boxes, final_scores, final_class_ids

    def process_result(self, result):
        boxes, scores, class_ids = self.custom_postprocess(result)
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            xyxy = box.tolist()
            centroid = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            detection = inference_pb2.Detection(
                label=self.class_names[str(class_id)],
                confidence=float(score),
                bbox=xyxy,
                centroid=centroid
            )
            detections.append(detection)
        return detections

    def Predict(self, request, context):
        try:
            camera_ip = request.image.camera_ip
            logger.info(f"Received a single inference request from camera IP: {camera_ip}")
            start_time = time.time()

            image = Image.open(io.BytesIO(request.image.image_data))
            image_np = np.array(image)

            results = self.model(image_np, conf=config['model']['confidence_threshold'], 
                                 classes=self.target_classes)
            detections = self.process_result(results[0])

            response = inference_pb2.PredictResponse(camera_ip=camera_ip, detections=detections)

            latency = time.time() - start_time
            logger.info(f"Single inference completed in {latency:.4f} seconds for camera IP: {camera_ip}")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return response

        except Exception as e:
            logger.error(f"Error during single inference for camera IP {camera_ip}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during inference: {str(e)}")
            return inference_pb2.PredictResponse()

    def BatchPredict(self, request, context):
        try:
            logger.info(f"Received a batch inference request with {len(request.images)} images")
            start_time = time.time()

            batch_response = inference_pb2.BatchPredictResponse()
            for image_data in request.images:
                camera_ip = image_data.camera_ip
                image = Image.open(io.BytesIO(image_data.image_data))
                image_np = np.array(image)

                results = self.model(image_np, conf=config['model']['confidence_threshold'], 
                                     classes=self.target_classes)
                detections = self.process_result(results[0])

                batch_response.results.append(inference_pb2.PredictResponse(camera_ip=camera_ip, detections=detections))

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
            camera_ip = request.image.camera_ip
            logger.info(f"Received a test inference request from camera IP: {camera_ip}")
            start_time = time.time()

            image = Image.open(io.BytesIO(request.image.image_data))
            image_np = np.array(image)
            # Test Predict must return all classes
            results = self.model(image_np, conf=config['model']['confidence_threshold'])
            result = results[0]

            detections = self.process_result(result)

            # Plot image with centroids
            plot_image = result.plot(boxes=True, conf=True, line_width=2)
            
            # Add centroids to the plot
            for detection in detections:
                centroid = detection.centroid
                cv2.circle(plot_image, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

            plot_image_bytes = io.BytesIO()
            Image.fromarray(plot_image).save(plot_image_bytes, format='PNG')

            response = inference_pb2.TestPredictResponse(
                camera_ip=camera_ip,
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
            logger.info(f"Test inference completed in {latency:.4f} seconds for camera IP: {camera_ip}")

            if config['metrics']['enabled']:
                update_inference_count()
                update_inference_latency(latency)

            return response

        except Exception as e:
            logger.error(f"Error during test inference for camera IP {camera_ip}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An error occurred during test inference: {str(e)}")
            return inference_pb2.TestPredictResponse()

def serve():
    logger.info("Starting server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                     options=[('grpc.max_receive_message_length', 40 * 1024 * 1024)])
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"{config['server']['host']}:{config['server']['port']}")
    server.start()
    logger.info(f"Server started on {config['server']['host']}:{config['server']['port']} version {config['server']['version']}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()