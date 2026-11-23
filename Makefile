.PHONY: help build run stop restart logs status compose-up compose-down dev-install dev test clean

APP_NAME = yolo26-vietnamese-money
CONTAINER_NAME = yolo26-money-app
PORT = 8000

help:
	@echo "=========================================================="
	@echo " YOLO26 Vietnamese Money Recognition - Automation Makefile"
	@echo "=========================================================="
	@echo "  make build         : Build Docker image"
	@echo "  make run           : Run Docker container with models mounted"
	@echo "  make stop          : Stop and remove running container"
	@echo "  make restart       : Restart container"
	@echo "  make logs          : Stream container logs"
	@echo "  make status        : Check container status"
	@echo "  make compose-up    : Launch using Docker Compose"
	@echo "  make compose-down  : Stop Docker Compose services"
	@echo "  make dev-install   : Install Python dependencies locally"
	@echo "  make dev           : Run local FastAPI dev server (port $(PORT))"
	@echo "  make test          : Run prediction smoke test"
	@echo "  make clean         : Remove temporary caches and runs"
	@echo "=========================================================="

build:
	@echo "[*] Building Docker image $(APP_NAME):latest..."
	docker build -t $(APP_NAME):latest .

run:
	@echo "[*] Running Docker container $(CONTAINER_NAME) on port $(PORT)..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		-v $$(pwd)/models:/app/models \
		--restart unless-stopped \
		$(APP_NAME):latest
	@echo ""
	@echo "[+] Container running successfully!"
	@echo "[+] Open Web UI:  http://localhost:$(PORT)"
	@echo "[+] Open API Doc: http://localhost:$(PORT)/docs"

stop:
	@echo "[*] Stopping container $(CONTAINER_NAME)..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	@echo "[+] Container stopped and removed."

restart: stop run

logs:
	docker logs -f $(CONTAINER_NAME)

status:
	docker ps -a --filter name=$(CONTAINER_NAME)

compose-up:
	docker compose up -d --build
	@echo "[+] Services started via Docker Compose! Open: http://localhost:$(PORT)"

compose-down:
	docker compose down

dev-install:
	pip install -r requirements.txt

dev:
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	python3 tests/test_detector.py

clean:
	rm -rf __pycache__ app/__pycache__ .pytest_cache runs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
