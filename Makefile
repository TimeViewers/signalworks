.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

export PIP_USE_PEP517=false
export PIPENV_VENV_IN_PROJECT=1

PYTHON:=$(shell which python3)
PIP:=$(PYTHON) -m pip
PIPENV:=$(shell which pipenv)
PRE_COMMIT:=$(PIPENV) run pre-commit
VENV:=$(shell $(PIPENV) --venv)

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
export PIPENV_IGNORE_VIRTUALENVS=1
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-precommit clean-mypy ## remove all build, test, coverage and Python artifacts

remove-venv:
	rm -rf $(VENV)

clean-precommit:
	pipenv run pre-commit clean

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-mypy:
	rm -rf .mypy_cache

clean-venv:
	@rm -rf $(VENV)

lint: ## check style with flake8
	$(PIPENV) run flake8 signalworks tests
	$(PIPENV) run mypy signalworks tests

test: ## run tests quickly with the default Python
	$(PIPENV) run python setup.py test

test-all: ## run tests on every Python version with tox
	$(PIPENV) run tox

coverage: ## check code coverage quickly with the default Python
	$(PIPENV) run coverage run --source signalworks -m pytest
	$(PIPENV) run coverage report -m
	$(PIPENV) run coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/signalworks.rst
	rm -f docs/modules.rst
	$(PIPENV) run sphinx-apidoc -o docs/ signalworks
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	$(PIPENV) run watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	$(PIPENV) run twine upload dist/*

dist: clean  ## builds source and wheel package
	$(PIPENV) run python setup.py sdist
	$(PIPENV) run python setup.py bdist_wheel
	ls -l dist

style: black isort

black:
	$(PIPENV) run black .

isort:
	$(PIPENV) run isort -y

format: style

install: clean remove-venv ## install the package to the active Python's site-packages
	$(PIPENV) install --dev --skip-lock .
	$(PIPENV) run pre-commit install
	$(PIPENV) run pre-commit autoupdate

lock:
	$(PIPENV) lock --clear

bump-minor:
	$(PIPENV) run bump2version minor

bump-patch:
	$(PIPENV) run bump2version patch

bump-major:
	$(PIPENV) run bump2version major
