from numpy.f2py import compile

# equivalent to np.maximum()
fmax = """
\tsubroutine fmax(x, dim, x_max)
\timplicit none
\tdouble precision, dimension(:), intent(in) :: x
\tinteger, intent(in) :: dim
\tdouble precision, intent(out) :: x_max
\tx_max = maxval(x, dim)
\treturn
\tend subroutine fmax
"""
compile(fmax, modulename="fortran_maximum", verbose=False)
exec(f"from fortran_maximum import fmax", globals())
fmax = globals().pop('fmax')

# equivalent to np.minimum()
fmin = """
\tsubroutine fmin(x, dim, x_min)
\timplicit none
\tdouble precision, dimension(:), intent(in) :: x
\tinteger, intent(in) :: dim
\tdouble precision, intent(out) :: x_min
\tx_min = minval(x, dim)
\treturn
\tend subroutine fmin
"""
compile(fmin, modulename="fortran_minimum", verbose=False)
exec(f"from fortran_minimum import fmin", globals())
fmin = globals().pop('fmin')

# equivalent to np.sum()
fsum = """
\tsubroutine fsum(x, dim, x_sum)
\timplicit none
\tdouble precision, dimension(:), intent(in) :: x
\tinteger, intent(in) :: dim
\tdouble precision, intent(out) :: x_sum
\tx_sum = sum(x, dim)
\treturn
\tend subroutine fsum
"""
compile(fsum, modulename="fortran_sum", verbose=False)
exec(f"from fortran_sum import fsum", globals())
fsum = globals().pop('fsum')

# equivalent to np.mean()
fmean = """
\tsubroutine fmean(x, dim, x_mean)
\timplicit none
\tdouble precision, dimension(:), intent(in) :: x
\tinteger, intent(in) :: dim
\tdouble precision, intent(out) :: x_mean
\tx_mean = sum(x, dim) / sum(shape(x))
\treturn
\tend subroutine fmean
"""
compile(fmean, modulename="fortran_mean", verbose=False)
exec(f"from fortran_mean import fmean", globals())
fmean = globals().pop('fmean')

# equivalent to np.matmul()
fmatmul = """
\tsubroutine fmatmul(x, y, z)
\timplicit none
\tdouble precision, dimension(:,:), intent(in) :: x, y
\tdouble precision, dimension(:,:), intent(out) :: z
\tz = matmul(x, y)
\treturn
\tend subroutine fmatmul
"""
compile(fmatmul, modulename="fortran_matmul", verbose=False)
exec(f"from fortran_matmul import fmatmul", globals())
fmatmul = globals().pop('fmatmul')
