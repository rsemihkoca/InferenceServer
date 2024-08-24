
## illegal instruction error: 

nano ~/.bashrc

export OPENBLAS_CORETYPE=ARMV8

## cuda available is false
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```