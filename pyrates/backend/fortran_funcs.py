from typing import Union


def get_fmean(out_shape: Union[tuple, str]):
    func = f"""
    \tfunction fmean(x,dim)
    \tdouble precision :: fmean{out_shape}
    \tdouble precision:: x(:)
    \tinteger, optional :: dim
    \tinteger :: d
    \tif (present(dim)) then
    \t  d = dim
    \t  fmean = sum(x,d) / sum(shape(x))
    \telse
    \t  fmean = sum(x) / sum(shape(x))
    \tendif
    \tend function
    """
    return func


def get_fsoftmax(out_shape: Union[tuple, str]):
    func = f"""
    \tfunction fsoftmax(x,dim)
    \tdouble precision :: fsoftmax{out_shape}
    \tdouble precision :: x(:), xsum
    \tinteger, optional :: dim
    \tinteger :: d, n, s
    \ts = shape(x)
    \tdo n=1:s
    \t  fsoftmax(n) = exp(x(n))
    \tend do
    \tif (present(dim)) then
    \t  d = dim
    \t  xsum = sum(x,d)
    \telse
    \t  xsum = sum(x)
    \tendif
    \tdo n=1:s
    \t  fsoftmax(n) = fsoftmax(n) / xsum
    \tend do
    \tend function
    """
    return func


def get_fsigmoid(out_shape: Union[tuple, str]):
    func = f"""
    \tfunction fsigmoid(x,scaling, steepness, offset)
    \tdouble precision :: fsigmoid{out_shape}
    \tdouble precision :: x(:), scaling, steepness, offset
    \tinteger :: n, s
    \ts = shape(x)
    \tdo n=1:s
    \t  fsigmoid(n) = scaling / (1 + exp(steepness*(offset-x(n)))
    \tend do
    \tend function
    """
    return func


funcs = {
    'mean': {'str': get_fmean, 'call': 'fmean'},
    'softmax': {'str': get_fsoftmax, 'call': 'fsoftmax'},
    'sigmoid': {'str': get_fsigmoid, 'call': 'fsigmoid'},
}


def get_fortran_func(fname: str, out_shape: Union[tuple, str]) -> tuple:
    return funcs[fname]['str'](out_shape), funcs[fname]['call']
