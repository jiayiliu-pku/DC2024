FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/automated-petct-lesion-segmentation  && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm weights/ /opt/algorithm/weights/
COPY --chown=algorithm:algorithm datacentric-challenge /opt/algorithm/datacentric-challenge/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict_seg.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict_cls.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils_img.py /opt/algorithm/
COPY --chown=algorithm:algorithm nnunet /opt/algorithm/nnunet 

COPY --chown=algorithm:algorithm test/input/images /input/images


# 添加清华源环境变量
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

WORKDIR /opt/algorithm/datacentric-challenge
RUN pip install .
WORKDIR /opt/algorithm

RUN python -m pip install --user -U pip && \
    python -m pip install --user -r requirements.txt
    
RUN pip install ultralytics==8.2.63
RUN pip uninstall -y opencv-python
RUN pip install opencv-python-headless

# RUN pip uninstall -y numpy
# RUN pip install numpy==1.26.4

ENTRYPOINT ["python", "-m", "process", "$0", "$@"]
