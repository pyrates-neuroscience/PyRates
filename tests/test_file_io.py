"""Test suit for saving PyRates objects to or loading them from file."""

__author__ = "Daniel Rose"
__status__ = "Development"

import filecmp
import os
import pathlib

import pytest


def setup_module():
    print("\n")
    print("=======================")
    print("| Test Suite File I/O |")
    print("=======================")


class FileCompareError(Exception):
    pass


def compare_files(filename1, filename2):
    """Helper function to compare files"""

    if os.path.getsize(filename1) == os.path.getsize(filename2):
        if filecmp.cmp(filename1, filename2, shallow=False):
            pass
        else:
            raise FileCompareError("Files have different content.")
    else:
        raise FileCompareError("Files are not of the same size.")


def get_parent_directory():
    """Helper function to get path of current file"""
    return pathlib.Path(__file__).parent.absolute()


def test_yaml_template():
    path = "model_templates.neural_mass_models.jansenrit.JRC"
    from pyrates import clear_frontend_caches
    from pyrates import save, CircuitTemplate
    clear_frontend_caches()

    template = CircuitTemplate.from_yaml(path)

    # configure filenames
    out_file = os.path.join(get_parent_directory(), "output", "jansen_rit_template.yaml")
    test_file = os.path.join(get_parent_directory(), "resources", "jansen_rit_template.yaml")

    template.to_yaml(out_file)
    # to update the reference dump, uncomment the following
    #save(template, test_file, filetype='yaml')

    compare_files(out_file, test_file)

    path = f"{out_file.split('.')[0]}/JRC"
    template = CircuitTemplate.from_yaml(path)
    assert template
