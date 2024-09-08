
## illegal instruction error: 

nano ~/.bashrc

export OPENBLAS_CORETYPE=ARMV8

## cuda available is false
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## install docker compose

https://medium.com/jetson-docs/docker-compose-v2-on-jetson-nano-91db0c02493c


## create grpc

python3 -m grpc_tools.protoc -I=proto --python_out=proto --grpc_python_out=proto proto/inference.proto

## compose restart

alias compose-restart='sudo docker image prune -f && sudo docker compose down -v --remove-orphans && sudo docker compose up --build -d && docker compose logs -f'

## ncdu not showing all files

sudo ncdu

## create env with python 3.8

conda create -n inference python=3.8

## NMS not working

RTDETR and YOLOv10 do not use NMS.
To address your concern about redundant detection boxes, it's important to note that while YOLOv10 aims to improve efficiency by eliminating the need for traditional NMS, it may still produce overlapping boxes in certain scenarios. This can happen due to the inherent nature of object detection models and the complexity of real-world scenes.
solution:
https://github.com/ultralytics/ultralytics/issues/13894