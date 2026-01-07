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
        extra_compile_args=['-O3', '-fopenmp'],
        extra_link_args=['-lgomp'],
    )
]

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="hilo",
        version="0.0.1",
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
            "hydra-core",
            "numpy",
        ],
    )
