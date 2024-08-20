from fastapi import FastAPI, File, UploadFile
from model import YOLOv10X
from config import load_config
from logging import getLogger
from metrics import setup_metrics, INFERENCE_TIME
import cv2
import numpy as np
import torch

app = FastAPI()
config = load_config()
logger = getLogger()
model = YOLOv10X(**config['model'])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with INFERENCE_TIME.time():
        result = model.inference(img)

    logger.info(f"Prediction made for image: {file.filename}")
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    setup_metrics(app)
    torch.set_num_threads(1)  # Limit CPU threads as we're using GPU
    uvicorn.run(app, host="0.0.0.0", port=config['server']['http_port'])