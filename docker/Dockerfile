FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt upgrade -y
RUN apt install python3-pip -y
RUN apt install vim -y
RUN apt install tmux -y
RUN apt install git -y

RUN pip install tqdm six wandb
RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install gymnax