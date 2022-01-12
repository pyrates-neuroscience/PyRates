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


@pytest.mark.skip
def test_pickle_template():
    path = "model_templates.neural_mass_models.jansenrit.JRC"
    from pyrates.frontend.template import from_yaml, clear_cache
    clear_cache()

    template = from_yaml(path)

    # include pickle protocol number in file name for version interoperability
    from pickle import HIGHEST_PROTOCOL
    ext = f"p{HIGHEST_PROTOCOL}"

    # configure filenames
    out_file = os.path.join(get_parent_directory(), "output", f"jansen_rit_template.{ext}")
    test_file = os.path.join(get_parent_directory(), "resources", f"jansen_rit_template.{ext}")

    from pyrates import save, circuit_from_pickle

    save(template, out_file, filetype='pickle')
    # to update the reference dump, uncomment the following
    # save(template, test_file, filetype='pickle')

    compare_files(out_file, test_file)

    template = circuit_from_pickle(out_file)
    assert template


@pytest.mark.skip
def test_yaml_template():
    path = "model_templates.jansen_rit.circuit.JansenRitCircuit"
    from pyrates.frontend.template import clear_cache
    from pyrates import save, circuit_from_yaml
    clear_cache()

    template = circuit_from_yaml(path)

    # include pickle protocol number in file name for version interoperability
    from pickle import HIGHEST_PROTOCOL
    ext = f"p{HIGHEST_PROTOCOL}"

    # configure filenames
    out_file = os.path.join(get_parent_directory(), "output", f"jansen_rit_template.{ext}")
    test_file = os.path.join(get_parent_directory(), "resources", f"jansen_rit_template.{ext}")

    save(template, out_file, filetype='yaml', template_name='jrc1')
    # to update the reference dump, uncomment the following
    # save(template, test_file, filetype='yaml', template_name='jrc1')

    compare_files(out_file, test_file)

    template = circuit_from_yaml(out_file)
    assert template
