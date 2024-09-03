
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

python3 -m grpc_tools.protoc -I=protos --python_out=. --grpc_python_out=. protos/inference.proto

## compose restart

alias compose-restart='sudo docker image prune -f && sudo docker compose down -v --remove-orphans && sudo docker compose up --build -d && docker compose logs -f'

## ncdu not showing all files

sudo ncdu

## create env with python 3.8

conda create -n inference python=3.8