PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME) python=$(PYTHON) -y

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements-dev.txt


format:
	black .
	isort .

triton:
	docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v $(PWD)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 \
		tritonserver --model-repository=/models