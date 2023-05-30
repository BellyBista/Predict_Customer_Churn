# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to identify customers that are most likely to churn based on credit card. This is a clean version of `churn_notebook.ipynb`. The completed project will include a Python package that follows coding (PEP8) and engineering best practices.

## Files and data description
1. ***churn_library.py***: 
- A library of functions to find customers who are likely to churn.
2. ***churn_script_logging_and_tests.py***:
- Contain unit tests for the *churn_library.py* functions.
- Log any errors and INFO messages.

## Running Files
1. Create environment:
```bash
conda create --name churn_predict python=3.6 
conda activate churn_predict 
```
Install the linter and auto-formatter: pip install pylint pip install autopep8

you can manually install dependencies as:
```bash
python -m pip install -r requirements_py3.6.txt

Run: python churn_library.py python_script_logging_and_tests.py

check the pylint score using the below: pylint churn_library.py pylint churn_script_logging_and_tests.py

To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below: autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py autopep8 --in-place --aggressive --aggressive churn_library.py