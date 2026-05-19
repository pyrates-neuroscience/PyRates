# -*- coding: utf-8 -*-
#
#
# PyRates software framework for flexible implementation of neural
# network models and simulations. See also:
# https://github.com/pyrates-neuroscience/PyRates
#
# Copyright (C) 2017-2018 the original authors (Richard Gast and
# Daniel Rose), the Max-Planck-Institute for Human Cognitive Brain
# Sciences ("MPI CBS") and contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Shared code-generation helpers for 1-indexed target languages.

JuliaBackend and MatlabBackend both target 1-indexed languages and share four
small helpers: scalar coercion in :meth:`get_var`, ``"a:b"`` string-range
parsing in :meth:`_process_idx`, ``apply=False`` offset-toggling in
:meth:`create_index_str`, and the ``**`` → ``^`` rewrite in
:meth:`expr_to_str`.  Previously MatlabBackend inherited from JuliaBackend
purely to pick these up, then used ``super(JuliaBackend, self).XXX`` to skip
the Julia-specific overrides — a brittle MRO trick flagged by
BACKEND_CONSISTENCY_REVIEW.md §4.7.

The mixin keeps the shared helpers in one place without binding Matlab to
Julia.  Each backend now inherits ``(OneBasedCodegenMixin, BaseBackend)`` so
``super()`` from a mixin method resolves to BaseBackend, regardless of which
language-specific subclass we are in.
"""

from typing import Tuple, Union, Optional, Dict
import numpy as np

from .computegraph import ComputeVar


class OneBasedCodegenMixin:
    """Helpers shared by 1-indexed code-generating backends."""

    # ------------------------------------------------------------------
    #  Variable conversion: scalars → Python int/float/complex
    # ------------------------------------------------------------------
    def get_var(self, v: ComputeVar):
        """Return the variable value in a form suitable for the host language.

        Multi-element arrays are passed through as numpy; 0-d (scalar)
        constants are coerced to a Python int/float/complex so the
        Julia / Matlab bridge can marshal them to native types directly.
        """
        v = super().get_var(v)
        dtype = v.dtype.name
        if sum(v.shape) > 0:
            return v
        if 'float' in dtype:
            return float(v)
        if 'complex' in dtype:
            return complex(np.real(v), np.imag(v))
        return int(v)

    # ------------------------------------------------------------------
    #  Range / index processing
    # ------------------------------------------------------------------
    def create_index_str(self, idx: Union[str, int, tuple], separator: str = ',',
                         apply: bool = True, **kwargs) -> Tuple[str, dict]:
        """``apply=False`` skips the language's start-index offset.

        The base routine derives all index expressions assuming the current
        ``self._start_idx``.  When the caller asks for the *unwrapped* index
        string (``apply=False``, used to splice an index into a generated
        expression rather than into bracket syntax) the offset is applied
        elsewhere, so we momentarily flip ``_start_idx`` to 0 to avoid
        double-counting.
        """
        if not apply:
            saved = self._start_idx
            self._start_idx = 0
            try:
                return super().create_index_str(idx, separator, apply, **kwargs)
            finally:
                self._start_idx = saved
        return super().create_index_str(idx, separator, apply, **kwargs)

    def _process_idx(self, idx: Union[Tuple[int, int], int, str, ComputeVar],
                     **kwargs) -> str:
        """Parse ``"a:b"`` string ranges with offset=0 then format the tuple.

        Without the temporary offset flip the inner ``int(self._process_idx(idx0))``
        would already see ``"{a + 1}"`` and double-count when the outer tuple
        formatter applies ``self._start_idx`` again.
        """
        if type(idx) is str and idx != ':' and ':' in idx:
            idx0, idx1 = idx.split(':')
            saved = self._start_idx
            self._start_idx = 0
            try:
                idx0 = int(self._process_idx(idx0))
                idx1 = int(self._process_idx(idx1))
            finally:
                self._start_idx = saved
            return self._process_idx((idx0, idx1))
        return super()._process_idx(idx=idx, **kwargs)

    # ------------------------------------------------------------------
    #  Expression rewrites
    # ------------------------------------------------------------------
    @staticmethod
    def expr_to_str(expr: str, args: tuple) -> str:
        """Rewrite the Python power operator ``**`` to the language native ``^``."""
        while '**' in expr:
            expr = expr.replace('**', '^')
        return expr
