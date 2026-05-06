# ============================================================
# Stage 1: Build the Go API binary (CGo enabled)
# ============================================================
FROM golang:1.25.5-bookworm AS builder

# Install XGBoost development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY api/go.mod api/go.sum ./
RUN go mod download

COPY api/*.go .

# Build with CGo enabled
RUN CGO_ENABLED=1 go build -o /server -ldflags="-s -w" .

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM debian:bookworm-slim

# Install XGBoost runtime library
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxgboost0 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy Go binary
COPY --from=builder /server /server

# Copy local model and config files directly from the host
COPY training/output/model.json /data/model.json
COPY resources/mcc_risk.json /data/mcc_risk.json
COPY resources/normalization.json /data/normalization.json
COPY training/output/centroids.bin /data/centroids.bin
COPY training/output/ivf_offsets.bin /data/ivf_offsets.bin
COPY training/output/ivf_vectors.bin /data/ivf_vectors.bin
COPY training/output/ivf_labels.bin /data/ivf_labels.bin

RUN mkdir -p /tmp/sockets && chmod 777 /tmp/sockets

EXPOSE 8080
CMD ["/server", "--model", "/data/model.json", "--port", "8080"]