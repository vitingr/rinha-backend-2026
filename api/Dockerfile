# ── Build stage ────────────────────────────────────────────────────────────────
FROM golang:1.25.5-alpine AS builder

WORKDIR /src
COPY go.mod .
COPY *.go .

# Build a fully static binary — no libc dependency, smallest possible image
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /fraud-api .

# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM scratch

# Copy binary
COPY --from=builder /fraud-api /fraud-api

# Copy reference data files
COPY resources/ /app/resources/

EXPOSE 8080

ENV PORT=8080
ENV REFERENCES_PATH=/app/resources/references.json.gz
# Tune GC: target 80% heap use — reduces GC frequency on a read-heavy workload
ENV GOGC=80
# Limit GC pause via soft memory limit (adjust per container memory limit)
ENV GOMEMLIMIT=130MiB

ENTRYPOINT ["/fraud-api"]
