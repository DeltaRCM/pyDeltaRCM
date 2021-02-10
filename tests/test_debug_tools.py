# unit tests for deltaRCM_tools.py

import pytest

import sys
import os
import numpy as np

# from pyDeltaRCM.model import DeltaModel

import matplotlib.pyplot as plt

from utilities import test_DeltaModel
# import utilities


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_iwalk(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
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


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_no_grid(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type', grid=False)
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_velocity(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.update()
    test_DeltaModel.show_attribute('ux')
    return plt.gcf()


def test_plot_domain_badtype(test_DeltaModel):
    with pytest.raises(TypeError):
        test_DeltaModel.show_attribute(42)


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_single_tuple(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4))
    return plt.gcf()


def test_plot_domain_cell_type_single_badtuple(test_DeltaModel):
    test_DeltaModel.show_attribute('cell_type')
    with pytest.raises(ValueError):
        test_DeltaModel.show_ind((3, 4, 5))


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_list_tuple(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([(3, 4), (3, 5), (4, 4)])
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_single_index(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind(15)
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_list_index(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([15, 16, 17, 19, 20])
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_list_mix_tuple_index(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind([(1, 5), (1, 6), 17, 19, 20])
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_multiple_index_calls(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4))
    test_DeltaModel.show_ind((3, 5))
    test_DeltaModel.show_ind((3, 6))
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_multiple_diff_args(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4), 'bs')
    test_DeltaModel.show_ind((3, 5), 'r*')
    test_DeltaModel.show_ind((3, 6), 'g^')
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_cell_type_multiple_diff_kwargs(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.show_attribute('cell_type')
    test_DeltaModel.show_ind((3, 4), marker='s', color='cyan')
    test_DeltaModel.show_ind((3, 5), markersize=12)
    test_DeltaModel.show_ind([(3, 4), (3, 5), (3, 6), 19, 20],
                             color='green', marker='.')
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_multiple_subplots(test_DeltaModel):
    test_DeltaModel.update()
    fig, ax = plt.subplots(1, 3, figsize=(6, 4))
    test_DeltaModel.show_attribute('eta', ax=ax[1])
    test_DeltaModel.show_attribute('cell_type', ax=ax[0])
    test_DeltaModel.show_attribute('ux', ax=ax[2])
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_plot_domain_withlabel(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    test_DeltaModel.update()
    # This is a weak test, but it triggers coverage of the label lines.
    test_DeltaModel.show_attribute('ux', label='')
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_show_line_pts(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    _arr = np.column_stack((np.arange(9), 50*np.ones((9,))))
    test_DeltaModel.show_line(_arr, ax=ax)  # autoreshaped by self
    test_DeltaModel.show_line(_arr.T, 'r-.', ax=ax)  # autoreshaped by self
    return plt.gcf()


@pytest.mark.mpl_image_compare(remove_text=True)
def test_show_line_set_points(test_DeltaModel):
    fig, ax = plt.subplots(figsize=(5, 4))
    np.random.seed(0)
    test_DeltaModel.free_surf_walk_inds = np.tile(
        np.arange(4, 60, step=10),
        (test_DeltaModel._Np_water, 1))
    _shape = test_DeltaModel.free_surf_walk_inds.shape
    test_DeltaModel.free_surf_walk_inds += np.random.randint(-1, 2,
                                                             size=_shape)
    test_DeltaModel.show_attribute('eta', ax=ax)
    test_DeltaModel.show_line(test_DeltaModel.free_surf_walk_inds,
                              ax=ax)
    return plt.gcf()
