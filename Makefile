.PHONY: install install-uv clean test

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e .

install-uv:
	uv venv
	uv pip install -e .

clean:
	rm -rf .venv
	rm -rf build
	rm -rf *.egg-info

test:
	. .venv/bin/activate && pytest
