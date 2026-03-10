
## Introduction


## News

## Installation

### Install from source code:

```
git clone https://github.com/EnVision-Research/DVD.git
cd DVD
conda create -n DVD python=3.10 -y 
conda activate dvd 
pip install -e .
```

### Install SageAttention (For Speedup):
```
pip install sageattention
```
### Download the checkpoint from Huggingface

```
mkdir ckpt
cd ckpt 
huggingface
```

If you encounter issues during installation, it may be caused by the packages we depend on. Please refer to the documentation of the package that caused the problem.

* [torch](https://pytorch.org/get-started/locally/)
* [sentencepiece](https://github.com/google/sentencepiece)
* [cmake](https://cmake.org)
* [cupy](https://docs.cupy.dev/en/stable/install.html)

## 3 Inference

### 3.1. For AIGC or Open World Evaluation (Stable Setting)
```
bash infer_bash/openworld.sh
```

### 3.2. For Academic Purpose (Paper Setting)

#### 3.2.1 Image Inference

For depth estimation, you can download the [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/) (depth) by the following commands (referred to Marigold).

Run the image inference script

```
bash infer_bash/image.sh
```

#### 3.2.2 Video Inference

Download the [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/), [Bonn Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [ScanNet Dataset](http://www.scan-net.org/).

Run the video inference script
```
bash infer_bash/video.sh
```



## Training 
