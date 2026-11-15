# METIS: one-command demo image (CPU only, no DNS data, no API key).
#
#   docker build -t metis .
#   docker run --rm -v "$PWD/demo_output:/out" --user "$(id -u):$(id -g)" metis
#
# Any other `metis` subcommand works too, e.g.
#   docker run --rm metis --help
#   docker run --rm -v /path/to/dns_data:/data -e METIS_DATA_ROOT=/data metis cases list
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/metis

# Dependencies first so code edits don't bust the layer cache.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[report]"

# Configs, the committed regime-v1 feature sample and frozen results.
COPY configs ./configs
COPY tests/data ./tests/data
COPY results ./results

# Runs as root by default so a bind-mounted /out is always writable; pass
# --user "$(id -u):$(id -g)" to get host-owned output files on Linux.
RUN mkdir -m 777 /out

ENTRYPOINT ["metis"]
CMD ["demo", "--out", "/out"]
