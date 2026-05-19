FROM python:3.11-bullseye

RUN mkdir -p /sandbox /tmp/sandbox_output /tmp/sandbox_plots \
    && pip install --upgrade pip \
    && pip install numpy pytest

WORKDIR /sandbox