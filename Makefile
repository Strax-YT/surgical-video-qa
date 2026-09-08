.PHONY: help install test lint run demo docker up down bench k8s-local clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s $$'\t'

install:  ## install deps into the active environment
	pip install -r requirements.txt

test:  ## run the suite (stub backend, no GPU or database needed)
	SVQA_BACKEND=stub python -m pytest

lint:  ## ruff check
	ruff check src tests

run:  ## serve the API in stub mode with reload
	SVQA_BACKEND=stub PYTHONPATH=src uvicorn svqa.api.main:app --reload --port 8000

demo:  ## ingest a generated sample video and ask it questions
	SVQA_BACKEND=stub PYTHONPATH=src python scripts/demo.py

bench:  ## benchmark model variants against a video
	SVQA_BACKEND=$(or $(SVQA_BACKEND),stub) PYTHONPATH=src python scripts/bench.py $(VIDEO)

docker:  ## build the image
	docker build -t surgical-video-qa:latest .

up:  ## start neo4j + api
	docker compose up --build

down:
	docker compose down -v

k8s-local:  ## deploy to a local kind cluster
	kind create cluster --name svqa || true
	docker build -t surgical-video-qa:latest .
	kind load docker-image surgical-video-qa:latest --name svqa
	kubectl apply -f k8s/
	kubectl rollout status deployment/svqa-api --timeout=180s

clean:
	rm -rf .pytest_cache .ruff_cache bench_results **/__pycache__
