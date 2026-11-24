.PHONY: install test lint classify regress all clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

classify:
	mlp-uci classify --out reports/beed_classification

regress:
	mlp-uci regress --out reports/actg175_regression

all: classify regress

clean:
	rm -rf reports/beed_classification reports/actg175_regression
