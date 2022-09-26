# Notes on Testing

### Static type checking

MyPy is used to statically check types. To test, if everything works out, run:

`MYPYPATH=./stubs/ mypy --strict-optional --ignore-missing-imports core`

If you get no output, all type checks are successful. Some issues are ignored using the comment tag

`# type: ignore`

These issues may be too complicated for mypy to recognise them properly - or too complicated to fix immediately, 
but might need fixing, nevertheless. 


### Running tests with py.test

We use `py.test` for testing.

Make sure you have `pytest` installed. 

Usage:
- from within PyCharm 
  - select py.test as default testing framework
  - right-click on tests and select "run py.test in tests"
- from the console
  - navigate to the PyRates base directory
  - run `pytest tests`

#### Structure of the tests folder

- `benchmarks.py`: scripts to be run to collect benchmark information (WIP)
- `resources` folder: represents data that is used in the tests (configuration or target data)
- scripts that contain tests for pyrates core code
