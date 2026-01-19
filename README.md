# HiLO
This is the official repo for "High-Level Object Fusion for Autonomous Driving using Transformers".

[arXiv](https://arxiv.org/abs/2506.02554) | [DOI](https://www.doi.org/10.1109/IV64158.2025.11097611) | [BibTeX](#citation)

## Installation
Tested on Ubuntu 22.04 with python 3.10.18, pytorch 2.7.1 and cuda 12.8.
See the exported conda environment at [environment.yaml](environment.yaml).
### With conda
```bash
conda create -n hilo python==3.10
conda activate hilo

conda install cuda -c nvidia/label/cuda-12.8.1

git clone https://github.com/rst-tu-dortmund/HiLO.git
cd HiLO

pip install -r requirements.txt
pip install -e .
```

### Other
1. Install CUDA on your machine or inside your environment [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) / [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).
2. Install requirements and HiLO:
```bash
pip install -r requirements.txt
pip install -e .
```

## Data
t.b.n.

## Results
| Source Domain | F1 <br /> @ 0.5 | F1 <br /> @ 0.5 | F1 <br /> @ 0.5 | mAP | mAP | mAP | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|           | urban |highway | combined | urban | highway | combined | |
| Urban     | 59.14 | 64.97 | 62.06 | 23.84 | 16.88 | 24.69 | [urban_hilo_max_F1](model_weights/urban/mean_F1_98.pth) |
| Highway   | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n |
| Combined  | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n | t.b.n |

## Getting Started
Define your environment as a new yaml config at [src/hilo/config/environment](src/hilo/config/environment).
Make sure to set all fields as defined in [base_environment.yaml](src/hilo/config/environment/base_environment.yaml).

## Usage
### Training
To train the model, see the following example command with experiment [train/hilo_urban](src/hilo/config/experiment/train/hilo_urban.yaml) which trains HiLO on urban data:
```bash
python src/train.py environment=YOUR_ENVIRONMENT +experiment=train/hilo_urban
```

### Evaluation
To evaluate the model, see the following example command with experiment [eval/urban_trained/on_urban](src/hilo/config/experiment/eval/urban_trained/on_urban.yaml) which evaluates the urban trained HiLO model on urban data:
```bash
python src/eval.py environment=YOUR_ENVIRONMENT +experiment=eval/urban_trained/on_urban
```
By default, this will evaluate the model on the validation split.
Add this argument to compute the metrics on the test split:
```bash
+split=test
```

# Citation
If you use this code for your research, please cite the following paper:
```bibtex
@INPROCEEDINGS{osterburg2025hilo,
  author={Osterburg, Timo and Albers, Franz and Diehl, Christopher and Pushparaj, Rajesh and Bertram, Torsten},
  booktitle={2025 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={HiLO: High-Level Object Fusion for Autonomous Driving Using Transformers}, 
  year={2025},
  volume={},
  number={},
  pages={209-214},
  doi={10.1109/IV64158.2025.11097611}}
```