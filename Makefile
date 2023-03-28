uid := $(shell id -u)
py_main := main.py

docker_image  := pytorch-split-transformer
docker_built  := .DOCKER
docker_run_it := docker run --rm \
                 -v $(abspath .):/app \
								 -it $(docker_image)

.PHONY: run
run: $(docker_built)
	python $(py_main)
#	$(docker_run_it)

$(docker_built): Dockerfile
	docker build . -t $(docker_image)
	touch $(docker_built)
