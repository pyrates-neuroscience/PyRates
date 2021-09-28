from typing import Union


# function for calculating the mean of an nd-array
def get_fmean(idx: int, out_shape: Union[tuple, str]) -> str:
    func = f"""
function fmean_{idx}(x,dim)

double precision :: fmean_{idx}{out_shape}
double precision:: x(:)
integer, optional :: dim
integer :: d

if (present(dim)) then
  d = dim
  fmean_{idx} = sum(x,d) / sum(shape(x))
else
  fmean_{idx} = sum(x) / sum(shape(x))
endif

end function fmean_{idx}
    """
    return func


# function for calculating the softmax transform of a 1d-array
def get_fsoftmax(idx: int, out_shape: Union[tuple, str]) -> str:
    func = f"""
function fsoftmax_{idx}(x)

double precision :: fsoftmax_{idx}{out_shape}
double precision :: x(:), xsum
integer :: d, n, s

s = shape(x)
do n=1,s
  fsoftmax_{idx}(n) = exp(x(n))
end do
xsum = sum(fsoftmax_{idx})
do n=1,s
  fsoftmax_{idx}(n) = fsoftmax_{idx}(n) / xsum
end do

end function fsoftmax_{idx}
    """
    return func


# function for calculating the logistic function of an nd-array
def get_fsigmoid(idx: int, out_shape: Union[tuple, str]) -> str:

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


# wrapper function for interpolating a 1d-array
def get_finterp(idx: int, out_shape: Union[tuple, str]) -> str:
    func = f"""
function finterp_{idx}(x,y,x_new)

implicit none 

double precision :: finterp_{idx}
double precision :: x(:), y(:), x_new
double precision :: x_inc
integer :: n, s

s = size(x) 

if (x_new < x(1)) then
  finterp_{idx} = y(1)
else if (x_new > x(s)) then
  finterp_{idx} = y(s)
else
  do n = 1, s 
    if (x(n) > x_new) exit 
  end do
  if (n == 1) then 
    finterp_{idx} = y(1)
  else if (n == s+1) then
    finterp_{idx} = y(s)
  else
    x_inc = (x_new - x(n-1)) / (x(n) - x(n-1))
    finterp_{idx} = y(n) + x_inc*(y(n) - y(n-1))
  end if 
end if 

end function finterp_{idx}
"""
    return func


fortran_identifiers = {
    'mean': {'str': get_fmean, 'call': 'fmean'},
    'softmax': {'str': get_fsoftmax, 'call': 'fsoftmax'},
    'sigmoid': {'str': get_fsigmoid, 'call': 'fsigmoid'},
    'interp': {'str': get_finterp, 'call': 'finterp'}
}


def get_fortran_func(fname: str, out_shape: Union[tuple, str], idx: int = 1) -> tuple:
    return fortran_identifiers[fname]['str'](idx, out_shape), f"{fortran_identifiers[fname]['call']}_{idx}"
