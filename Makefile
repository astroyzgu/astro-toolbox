.PHONY: test build
all: install test

install: 
	pip install . 

clean: 
	rm -rf mypackage.egg-info 
	rm -rf build
	rm -rf dist
test: 
	pytest