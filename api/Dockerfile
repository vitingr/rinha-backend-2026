FROM golang:1.25.5-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./

RUN go mod download -x

COPY *.go .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -trimpath -o /fraud-api .

FROM scratch

COPY --from=builder /fraud-api /fraud-api
COPY resources/ /app/resources/

EXPOSE 9999

ENV PORT=9999
ENV REFERENCES_PATH=/app/resources/references.json.gz
ENV GOGC=25
ENV GOMEMLIMIT=130MiB
ENV GOMAXPROCS=2

ENTRYPOINT ["/fraud-api"]