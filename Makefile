PYTHON ?= python

.PHONY: install install-locked data reproduce reproduce-full verify test clean

install:
	$(PYTHON) -m pip install -e .

install-locked:
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(PYTHON) scripts/download_data.py

reproduce:
	$(PYTHON) scripts/reproduce.py --profile core

reproduce-full:
	$(PYTHON) scripts/reproduce.py --profile full

verify:
	$(PYTHON) scripts/verify_results.py

test:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	$(PYTHON) scripts/clean_generated.py
