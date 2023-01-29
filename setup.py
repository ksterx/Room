import os

from setuptools import setup

LIB_NAME = "room"
root_dir = os.path.dirname(os.path.realpath(__file__))
INSTALL_REQUIRES = [
    "gym",
    "torch",
    "hydra-core>=1.2",
    "ray[default]",
]


setup(
    name=LIB_NAME,
    author="ksterx",
    version=open(f"{root_dir}/{LIB_NAME}/version.txt").read(),
    python_requires=">=3.8.*",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
    },
    package_dir={"": "room"},
)
