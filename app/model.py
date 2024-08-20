import torch
import numpy as np
import cv2
from queue import Queue
from threading import Thread


class YOLOv10X:
    def __init__(self, model_path, input_size, confidence_threshold, nms_threshold, batch_size=4):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)

        self.queue = Queue(maxsize=64)
        self.result_queue = Queue()
        self._start_worker()

    def _load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        return torch.jit.script(model)  # Use TorchScript

    def preprocess(self, image):
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    def postprocess(self, output):
        # Implement post-processing logic here (NMS, etc.)
        # This is a placeholder and needs to be implemented based on YOLOv10x output format
        pass

    @torch.no_grad()
    def _inference(self, batch):
        output = self.model(batch)
        return [self.postprocess(out) for out in output]

    def _worker(self):
        while True:
            batch = []
            for _ in range(self.batch_size):
                item = self.queue.get()
                if item is None:
                    break
                batch.append(item)

            if not batch:
                break

            input_batch = torch.cat(batch, dim=0)
            results = self._inference(input_batch)

            for result in results:
                self.result_queue.put(result)

    def _start_worker(self):
        Thread(target=self._worker, daemon=True).start()

    def inference(self, image):
        preprocessed = self.preprocess(image)
        self.queue.put(preprocessed)
        return self.result_queue.get()

    def __del__(self):
        self.queue.put(None)  # Signal the worker to stop