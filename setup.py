import pybind11, torch
from pybind11.setup_helpers import Pybind11Extension
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)

import setuptools

ext_modules = [
    CppExtension(
        name="hilo.evaluation.metrics.iou_bev_cpp",
        sources=["src/hilo/evaluation/metrics/cpp/iou_bev.cpp"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-lgomp"],
    )
]

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="hilo",
        version="1.0.0",
        author="Timo Osterburg",
        author_email="timo.osterburg@tu-dortmund.de",
        description="Implementation of HiLO: High-Level Object Fusion for Autonomous Driving using Transformers",
        long_description=long_description,
        long_description_content_type="text/markdown",
        project_urls={},
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "src"},
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
        packages=setuptools.find_packages(where="src"),
        python_requires=">=3.10",
        install_requires=[
            "einops==0.8.1",
            "hydra-core==1.3.2",
            "kaleido==1.1.0",
            "numpy==2.2.6",
            "omegaconf==2.3.0",
            "omegaconf_resolver @ git+https://github.com/rst-tu-dortmund/omegaconf_resolver.git",
            "opencv-python==4.12.0.88",
            "pandas==2.3.1",
            "pillow==11.3.0",
            "plotly==6.3.1",
            "protobuf==6.31.1",
            "pybind11==3.0.1",
            "pytest==8.4.2",
            "pytest-cov==7.0.0",
            "pytest-timeout==2.4.0",
            "scipy==1.15.3",
            "shapely==2.1.2",
            "torch==2.7.1",
            "torchaudio==2.7.1",
            "torchvision==0.22.1",
            "tqdm==4.67.1",
            "wandb==0.21.0",
        ],
    )
