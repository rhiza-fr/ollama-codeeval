FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /sandbox /tmp/sandbox_output /tmp/sandbox_plots \
    && pip install --upgrade pip \
    && pip install numpy pandas matplotlib pillow seaborn scikit-learn scipy scikit-image plotly pytest ruff

WORKDIR /sandbox