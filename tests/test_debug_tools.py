# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

# from pyDeltaRCM.model import DeltaModel

import matplotlib.pyplot as plt

from utilities import test_DeltaModel
# import utilities


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_iwalk(test_DeltaModel):
    test_DeltaModel.show_attribute('iwalk')
    return plt.gcf()


def test_plot_attribute_bad_shape_1d(test_DeltaModel):
    with pytest.raises(ValueError):
        test_DeltaModel.show_attribute('free_surf_flag')


def test_plot_attribute_bad_shape_3d(test_DeltaModel):
    # not sure if there are any 3d arrays actually, so just make a fake one
    test_DeltaModel.threedeearray = np.zeros((10, 10, 2))
    with pytest.raises(ValueError):
        test_DeltaModel.show_attribute('threedeearray')


def test_plot_attribute_missing_unknown(test_DeltaModel):
    with pytest.raises(AttributeError):
        test_DeltaModel.show_attribute('FAKE_NAME_ATTRIBUTE')


def test_plot_attribute_numeric_bad_type(test_DeltaModel):
    with pytest.raises(TypeError):
        test_DeltaModel.show_attribute('dx')
    with pytest.raises(TypeError):
        test_DeltaModel.show_attribute('strata_sand_frac')


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_no_grid(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type', grid=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_velocity(test_DeltaModel):
    test_DeltaModel.update()
    test_DeltaModel.show_attribute('ux')
    return plt.gcf()


def test_plot_domain_badtype(test_DeltaModel):
    test_DeltaModel.update()
    with pytest.raises(TypeError):
        test_DeltaModel.show_attribute(42)


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_single_tuple(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4))
    return plt.gcf()


def test_plot_domain_cell_type_single_badtuple(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    with pytest.raises(ValueError):
        test_DeltaModel.show_ind((3, 4, 5))


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_list_tuple(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([(3, 4), (3, 5), (4, 4)])
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_single_index(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind(15)
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_list_index(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([15, 16, 17, 19, 20])
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_list_mix_tuple_index(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([(1, 5), (1, 6), 17, 19, 20])
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_multiple_index_calls(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4))
    test_DeltaModel.show_ind((3, 5))
    test_DeltaModel.show_ind((3, 6))
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_multiple_diff_args(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4), 'bs')
    test_DeltaModel.show_ind((3, 5), 'r*')
    test_DeltaModel.show_ind((3, 6), 'g^')
    return plt.gcf()


@pytest.mark.mpl_image_compare()
def test_plot_domain_cell_type_multiple_diff_kwargs(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4), marker='s', color='cyan')
    test_DeltaModel.show_ind((3, 5), markersize=12)
    test_DeltaModel.show_ind([(3, 4), (3, 5), (3, 6), 19, 20],
                             color='green', marker='.')
    return plt.gcf()
