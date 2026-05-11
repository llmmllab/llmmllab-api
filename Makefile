PORT ?= 9999
DOCKER_IMAGE ?= llmmllab-api
DOCKER_TAG ?= latest
HELM_KUBECONTEXT ?= lsnet
REGISTRY ?= 192.168.0.71:31500

export HELM_KUBECONTEXT

# Development
start: db-up
	@echo "Starting API server on port $(PORT)..."
	@export $(shell grep -v '^#' .env 2>/dev/null | xargs 2>/dev/null) && \
	uv run python -m uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload

start-docker:
	@echo "Starting API server in Docker..."
	@bash ./run.sh

# Local database (TimescaleDB via Docker)
db-up:
	@echo "Starting local PostgreSQL..."
	@docker compose up -d --wait postgres

db-down:
	@echo "Stopping local PostgreSQL..."
	@docker compose down

db-reset: db-down
	@echo "Removing PostgreSQL volume and restarting..."
	@docker volume rm -f llmmllab-api_llmmllab-postgres-data 2>/dev/null || true
	@$(MAKE) db-up

# Testing
test:
	@echo "Running tests..."
	@uv run pytest test/

test-unit:
	@echo "Running unit tests..."
	@uv run pytest test/unit/

# Validation
validate:
	@echo "Validating Python syntax..."
	@python -m compileall -q -x '(venv|\.venv)' .
	@echo "Validation complete!"

# Cleanup
clean:
	@echo "Cleaning artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Artifacts cleaned."

# Docker — multi-arch build and push
# Buildx --push doesn't support insecure registries, so we build each arch
# with --load, push with docker push, then create a manifest list.
.PHONY: docker-build docker-push test

docker-build:
	@echo "Building amd64 image..."
	@docker buildx build --builder multibuilder --platform linux/amd64 -t $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)-amd64 --load .
	@echo "Building arm64 image..."
	@docker buildx build --builder multibuilder --platform linux/arm64 -t $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)-arm64 --load .
	@echo "Pushing amd64 image..."
	@docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)-amd64
	@echo "Pushing arm64 image..."
	@docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)-arm64
	@echo "Creating multi-arch manifest..."
	@python3 scripts/push_manifest.py \
		--registry $(REGISTRY) \
		--repo $(DOCKER_IMAGE) \
		--tag $(DOCKER_TAG) \
		--user bcf186aef4ebc292 \
		--password b6c98846d1e66359903a2137

docker-push:
	@echo "Pushing Docker image..."
	@docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# Kubernetes
k8s-apply:
	@echo "Applying Kubernetes manifests..."
	@chmod +x k8s/apply.sh
	@DOCKER_TAG=$(DOCKER_TAG) ./k8s/apply.sh

deploy: docker-build k8s-apply
	@echo "Deployment complete!"

# Sync (k8s dev mode)
sync:
	@echo "Syncing code to k8s node..."
	@chmod +x sync-code.sh
	@./sync-code.sh

sync-watch:
	@echo "Watching for changes and syncing..."
	@chmod +x sync-code.sh
	@./sync-code.sh -w
