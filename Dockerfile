FROM python:3.8.18-bullseye
# RUN sed -i 's/deb.debian.org/ftp.cn.debian.org/g' /etc/apt/sources.list
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y libhdf5-dev
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install evaluate
RUN pip install nltk
RUN pip install --upgrade pip
RUN pip install datasets==1.17.0
RUN pip install argparse
RUN pip install tensorboard==2.10.0
RUN pip install rouge_score
RUN pip install deepspeed==0.10.0
RUN pip install peft==0.5.0
RUN pip install tqdm
RUN pip install torchtyping
RUN pip install rouge_score
RUN pip install wandb==0.12.10
RUN pip install sentencepiece==0.1.96
RUN pip install fairscale==0.4.5
RUN pip install ipython
RUN pip install transformers==4.23.1
RUN pip install accelerate==0.16.0
RUN pip install protobuf==3.19.0
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"