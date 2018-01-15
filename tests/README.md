# Notes on Testing

### Static type checking

MyPy is used to statically check types. To test, if everything works out, run:

`MYPYPATH=./stubs/ mypy --strict-optional --ignore-missing-imports core`

If you get no output, all type checks are successful. Some issues are ignored using the comment tag

`# type: ignore`

These issues may be too complicated for mypy to recognise them properly - or too complicated to fix immediately, 
but might need fixing, nevertheless. 

