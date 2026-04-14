FROM python:3.11-slim

WORKDIR /workspace

# system deps (needed for sphinx-multiversion + git-based builds)
RUN apt-get update && apt-get install -y \
    git \
    make \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY docs/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# default working dir for sphinx
WORKDIR /workspace/docs

# default build (can be overridden by compose)
CMD ["sphinx-build", "-W", ".", "_build/current"]
