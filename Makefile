.PHONY: setup db-up db-down init-db fetch-data train run pipeline test

setup:
	pip install -r requirements.txt

db-up:
	docker-compose up -d postgres

db-down:
	docker-compose down

init-db:
	python scripts/init_db.py

fetch-data:
	python scripts/fetch_historical.py

train:
	python scripts/train_model.py

run:
	uvicorn backend.app.main:app --reload --port 8000

pipeline:
	python scripts/run_daily_pipeline.py

test:
	pytest tests/ -v

# Full first-time setup sequence
bootstrap: db-up init-db fetch-data train
	@echo "Bootstrap complete — run 'make run' to start the API"
