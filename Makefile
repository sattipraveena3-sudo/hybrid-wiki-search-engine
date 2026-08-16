PYTHON ?= python3

.PHONY: install test run smoke docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

smoke:
	curl -fsS http://localhost:8000/health
	curl -fsS http://localhost:8000/ready
	curl -fsS 'http://localhost:8000/api/search?q=hybrid%20search&mode=hybrid'
	curl -fsS http://localhost:8000/metrics

docker-up:
	docker compose up --build

docker-down:
	docker compose down
