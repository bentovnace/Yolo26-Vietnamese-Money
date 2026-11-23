.PHONY: help build run stop restart logs status compose-up compose-down dev-install dev test hf-upload render-status render-deploy render-health render-open clean

APP_NAME = yolo26-vietnamese-money
CONTAINER_NAME = yolo26-money-app
PORT = 8000
RENDER_URL = https://yolo26-vietnamese-money.onrender.com
RENDER_SERVICE_ID = srv-dc26un2jnfac7391uht0

help:
	@echo "=========================================================="
	@echo " YOLO26 Vietnamese Money Recognition - Automation Makefile"
	@echo "=========================================================="
	@echo "  make dev           : Run local dev server with Render Cloud connection"
	@echo "  make build         : Build Docker image"
	@echo "  make run           : Run Docker container with models mounted"
	@echo "  make stop          : Stop and remove running container"
	@echo "  make restart       : Restart container"
	@echo "  make logs          : Stream container logs"
	@echo "  make status        : Check container status"
	@echo "  make compose-up    : Launch using Docker Compose"
	@echo "  make compose-down  : Stop Docker Compose services"
	@echo "  make dev-install   : Install Python dependencies locally"
	@echo "  make test          : Run prediction smoke test"
	@echo "  make hf-upload     : Upload weights to Hugging Face Model Hub"
	@echo "  make render-status : Check Render Cloud deployment status"
	@echo "  make render-deploy : Trigger new deploy on Render"
	@echo "  make render-health : Check Render Cloud healthcheck"
	@echo "  make render-open   : Display Render Cloud URLs"
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
	@echo "=========================================================="
	@echo "🚀 Starting YOLO26 Dev Server & Connecting to Render Cloud"
	@echo "=========================================================="
	@echo "🏠 Local Web UI:    http://localhost:$(PORT)"
	@echo "☁️ Render Cloud:    $(RENDER_URL)"
	@echo "📖 Local API Docs:  http://localhost:$(PORT)/docs"
	@echo "📖 Render API Docs: $(RENDER_URL)/docs"
	@echo "💡 Giao diện Web hỗ trợ chọn Backend: Local hoặc Render!"
	@echo "=========================================================="
	uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

test:
	python3 tests/test_detector.py

hf-upload:
	python3 scripts/upload_to_hf.py

render-status:
	@curl -s -H "Authorization: Bearer rnd_f0dYD05A1TLu4XAofxBnWVOC5ZQF" \
		https://api.render.com/v1/services/$(RENDER_SERVICE_ID)/deploys?limit=1 | grep -o '"status":"[^"]*"' || echo "Check Render dashboard"

render-health:
	@curl -s $(RENDER_URL)/health | python3 -m json.tool

render-deploy:
	@curl -s -X POST -H "Authorization: Bearer rnd_f0dYD05A1TLu4XAofxBnWVOC5ZQF" \
		-H "Accept: application/json" -H "Content-Type: application/json" \
		-d '{"clearCache":"clear"}' \
		https://api.render.com/v1/services/$(RENDER_SERVICE_ID)/deploys

render-open:
	@echo "☁️ Render Web UI:     $(RENDER_URL)"
	@echo "📖 Render Swagger UI: $(RENDER_URL)/docs"
	@echo "🩺 Render Healthcheck:$(RENDER_URL)/health"

clean:
	rm -rf __pycache__ app/__pycache__ .pytest_cache runs/
	find . -type d -name "__pycache__" -exec rm -rf {} +
