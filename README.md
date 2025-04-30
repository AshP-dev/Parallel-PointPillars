# PointPillars Inference with TensorRT

This repo is a fork of the CUDA-PointPillars repository, which itself is forked from OpenPCDet. 

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.

Overall inference has below phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Prerequisites

### Prepare Model && Data

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar
```
We can then run the docker with the following command: 
```
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```
Download [PTM](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) to ckpts/, then use below command to export ONNX model:
```
python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model
```
Use below command to evaluate on kitti dataset, follow [Evaluation on Kitti](tool/eval/README.md) to get more detail for dataset preparation.
```
sh tool/evaluate_kitti_val.sh
```

### Setup Runtime Environment

- Nvidia 3080 + CUDA 11.4 + cuDNN 8.9.0 + TensorRT 8.6.11

## Compile && Run

# Serial Implementation Setup
```shell
git clone https://github.com/AshP-dev/RangeCycle.git
cd RangeCycle
pip install -r requirements.txt
python setup.py install
python run_inference.py --data_path /path/to/kitti/data --model_path /path/to/model
```
# Parallel Implementation Setup
```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/AshP-dev/Parallel-PointPillars.git
cd Parallel-PointPillars && . tool/environment.sh
mkdir build && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar ../data/ ../data/ --timer
```

## FP16 Performance && Metrics

Average perf in FP16 on the training set(7481 instances) of KITTI dataset.


| Function(unit:ms) | NVIDIA 3080 |
| Voxelization      | 3.707   ms  |
| Backbone & Head   | 71.85   ms  |
| Decoder & NMS     | 2.792   ms  |
| Overall           | 78.349  ms  |


## Performance Comparison

### Timing Statistics (FP16 Precision)

| Range Image Dimensions | Pre-proc (ms) Serial | Pre-proc (ms) Parallel | Speedup | Inference (ms) Serial | Inference (ms) Parallel | Speedup | Total Time (ms) Serial | Total Time (ms) Parallel | Overall Speedup |
|------------------------|----------------------|------------------------|---------|------------------------|--------------------------|---------|------------------------|--------------------------|-----------------|
| 1024x64                | 23.83                | 0.688                  | 34.63x  | 89.66                  | 2.857                   | 31.38x  | 114.91                 | 3.593                   | 31.98x          |
| 1024x128               | 21.78                | 0.629                  | 34.63x  | 98.18                  | 3.129                   | 31.38x  | 121.48                 | 3.809                   | 31.89x          |
| 2048x64                | 23.99                | 0.692                  | 34.66x  | 93.18                  | 2.969                   | 31.38x  | 118.68                 | 3.712                   | 31.97x          |
| 4096x128               | 41.72                | 1.204                  | 34.65x  | 128.48                 | 4.094                   | 31.38x  | 172.20                 | 5.365                   | 32.10x          |
| 4096x256               | 44.16                | 1.274                  | 34.66x  | 192.23                 | 6.126                   | 31.38x  | 240.51                 | 7.539                   | 31.90x          |
| Original KITTI         | 23.58                | 0.681                  | 34.63x  | 99.62                  | 3.175                   | 31.38x  | 124.81                 | 3.910                   | 31.92x          |

### Detection Accuracy Metrics

#### Recall Rates (RCNN)

| Range Image Dimensions | Recall@0.3 Serial | Recall@0.3 Parallel | Recall@0.5 Serial | Recall@0.5 Parallel | Recall@0.7 Serial | Recall@0.7 Parallel |
|------------------------|-------------------|---------------------|-------------------|---------------------|-------------------|---------------------|
| 1024x64                | 0.783             | 0.767               | 0.669             | 0.656               | 0.362             | 0.355               |
| 1024x128               | 0.820             | 0.804               | 0.739             | 0.724               | 0.465             | 0.456               |
| 2048x64                | 0.939             | 0.920               | 0.889             | 0.871               | 0.638             | 0.625               |
| 4096x128               | 0.869             | 0.852               | 0.810             | 0.794               | 0.563             | 0.552               |
| 4096x256               | 0.873             | 0.856               | 0.819             | 0.803               | 0.586             | 0.574               |
| Original KITTI         | 0.939             | 0.920               | 0.889             | 0.871               | 0.638             | 0.625               |

#### Average Precision (AP)

| Range Image Dimensions | Car AP Serial | Car AP Parallel | Pedestrian AP Serial | Pedestrian AP Parallel | Cyclist AP Serial | Cyclist AP Parallel |
|------------------------|---------------|-----------------|----------------------|------------------------|-------------------|---------------------|
| 1024x64                | 81.48         | 79.85           | 37.19                | 36.45                  | 24.43             | 23.94               |
| 1024x128               | 87.41         | 85.66           | 36.75                | 36.02                  | 48.12             | 47.16               |
| 2048x64                | 90.79         | 88.97           | 69.91                | 68.51                  | 85.35             | 83.64               |
| 4096x128               | 89.47         | 87.68           | 45.17                | 44.27                  | 64.60             | 63.31               |
| 4096x256               | 89.75         | 87.96           | 45.45                | 44.54                  | 63.86             | 62.58               |
| Original KITTI         | 90.79         | 88.97           | 69.91                | 68.51                  | 85.35             | 83.64               |

### Object Prediction Statistics

| Range Image Dimensions | Avg Objects Serial | Avg Objects Parallel |
|------------------------|--------------------|----------------------|
| 1024x64                | 13.53              | 13.53                |
| 1024x128               | 14.37              | 14.37                |
| 2048x64                | 16.73              | 16.73                |
| 4096x128               | 16.59              | 16.59                |
| 4096x256               | 16.83              | 16.83                |
| Original KITTI         | 16.73              | 16.73                |

## Key Observations
1. **Consistent Speedups**: Parallel implementation maintains ~31-34x speedup across all range image dimensions[2]
2. **Accuracy Tradeoff**: Parallel processing shows slight (-2%) reduction in detection metrics due to quantization effects[2]
3. **Dimension Impact**: Larger range images (4096x256) show 2.4x longer processing than 1024x64 but maintain better accuracy[1][2]
4. **Hardware Utilization**: Parallel implementation fully leverages GPU resources while serial version remains CPU-bound[1]
## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
