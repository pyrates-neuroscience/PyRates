from typing import Union


def get_fmean(out_shape: Union[tuple, str]):
    func = f"""
function fmean(x,dim)

double precision :: fmean{out_shape}
double precision:: x(:)
integer, optional :: dim
integer :: d

if (present(dim)) then
  d = dim
  fmean = sum(x,d) / sum(shape(x))
else
  fmean = sum(x) / sum(shape(x))
endif

end function
    """
    return func


def get_fsoftmax(out_shape: Union[tuple, str]):
    func = f"""
function fsoftmax(x,dim)

double precision :: fsoftmax{out_shape}
double precision :: x(:), xsum
integer, optional :: dim
integer :: d, n, s

s = shape(x)
do n=1,s
  fsoftmax(n) = exp(x(n))
end do
if (present(dim)) then
  d = dim
  xsum = sum(x,d)
else
  xsum = sum(x)
endif
do n=1,s
  fsoftmax(n) = fsoftmax(n) / xsum
end do

end function
    """
    return func


def get_fsigmoid(idx: int, out_shape: Union[tuple, str]):

    sigmoid_n = f"s = shape(x)\ndo n=1,s\n  fsigmoid_{idx}(n) = 1 / (1 + exp(-x(n)))\nend do"
    sigmoid_0 = f"fsigmoid_{idx} = 1 / (1 + exp(-x))"

    func = f"""
function fsigmoid_{idx}(x)

implicit none

double precision :: fsigmoid_{idx}{out_shape}
double precision :: x{'(:)' if len(out_shape) > 0 else ''}
{"integer :: n, s" if len(out_shape) > 0 else ""}

{sigmoid_n if len(out_shape) > 0 else sigmoid_0}

end function fsigmoid_{idx}
    """
    return func


fortran_identifiers = {
    'mean': {'str': get_fmean, 'call': 'fmean'},
    'softmax': {'str': get_fsoftmax, 'call': 'fsoftmax'},
    'sigmoid': {'str': get_fsigmoid, 'call': 'fsigmoid'},
}


def get_fortran_func(fname: str, out_shape: Union[tuple, str], idx: int = 1) -> tuple:
    return fortran_identifiers[fname]['str'](idx, out_shape), f"{fortran_identifiers[fname]['call']}_{idx}"
