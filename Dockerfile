# Multi-stage build: artifacts (Python) -> server (Go) -> web (Node) -> small runtime image.
FROM python:3.12-slim AS ml
WORKDIR /src
COPY ml/ ml/
RUN pip install --no-cache-dir -r ml/requirements.txt && cd ml && python run_pipeline.py

FROM golang:1.24 AS go
WORKDIR /src/serving
COPY serving/ .
RUN CGO_ENABLED=0 go build -o /out/muse-server ./cmd/server

FROM node:22-slim AS web
WORKDIR /src/web
COPY web/ .
RUN npm install && npm run build

FROM gcr.io/distroless/static-debian12
COPY --from=go /out/muse-server /muse-server
COPY --from=ml /src/artifacts /artifacts
COPY --from=web /src/web/dist /web
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/muse-server", "-artifacts", "/artifacts", "-static", "/web", "-addr", ":8080"]
