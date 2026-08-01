.PHONY: test overfit leakage clean

test:
	uv run pytest -q

overfit:
	uv run python bench/overfit.py

leakage:
	uv run python bench/leakage.py

clean:
	rm -rf .pytest_cache **/__pycache__
