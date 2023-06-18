# Occupancy Network Docker Container

## Build

To build the docker container:
```bash
docker build -t octran:v1 .
```
## Run

Once you have built the docker container use the following command to start it up:
```bash
docker run --gpus all -it --rm -p 8888:8888 -v $PWD:/OCTraN octran:v1
```

Once the container is up and running use the following code to launch jupyter notebooks:
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```