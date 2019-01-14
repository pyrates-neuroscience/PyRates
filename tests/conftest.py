
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
# 
# CITATION:
# 
# Richard Gast and Daniel Rose et. al. in preparation
""" pytest configuration file based on example from https://github.com/dropbox/pyannotate
This file configures py.test so that pyannotate is used to collect types in the tested module.
The results are saved as type_info.json.

Note that only files are checked for types that are in the current working directory. In order to check other files,
move the test files to the respective folder.
"""

# Configuration for pytest to automatically collect types.
# Thanks to Guilherme Salgado.

# import pytest
# from pyannotate_runtime import collect_types
#
# collect_types.init_types_collection()
#
#
# @pytest.fixture(autouse=True)
# def collect_types_fixture():
#     collect_types.resume()
#     yield
#     collect_types.pause()
#
#
# def pytest_sessionfinish(session, exitstatus):
#     collect_types.dump_stats("type_info.json")
