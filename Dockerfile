FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL version="0.0.2"

RUN apt update && apt install -y \
    tmux \
    htop \
    nano \
    vim \
    zip \
    git \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'set -g mouse on' > ~/.tmux.conf

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    numpy \
    wandb \
    lightning \
    scikit-learn \
    pyproj \
    seaborn \
    ipykernel \
    captum \
    xarray \
    timm \
    torchgeo \
    black \
    isort \
    flake8 \
    pytest \
    pytest-cov \
    PyYAML \
    scipy \
    requests \
    pyarrow \
    shapely \
    cartopy \
    h5netcdf \
    netCDF4 \
    "ocean-taco[hf]"

ENV workdir=/home/user
WORKDIR ${workdir}

CMD ["bash"]
