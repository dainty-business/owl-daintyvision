FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS owl_daintyvision
# cuda12.6 is what the Grand Challenge T4 instances need (official OWL starter kit)

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/mpl

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user app.py inference.py owl_wsi.py owl_mil.py report.py /opt/app/

# Grand Challenge detects the invoke API from this label
LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
