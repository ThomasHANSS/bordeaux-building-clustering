.PHONY: setup data features select cluster evaluate map all clean test

setup:
	pip install -e ".[dev]"

data:
	python -m src.data_loader

features:
	python -m src.features

select:
	python -m src.feature_selection

cluster:
	python -m src.clustering

evaluate:
	python -m src.evaluation

map:
	python -m src.mapping

all: data features select cluster evaluate map

clean:
	rm -rf outputs/maps/* outputs/figures/*
	rm -f data/processed/*.geoparquet data/processed/*.parquet

test:
	pytest tests/ -v
