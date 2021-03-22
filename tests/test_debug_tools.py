import pytest

import numpy as np
import matplotlib.pyplot as plt

from pyDeltaRCM.model import DeltaModel
from . import utilities


class TestShowAttributeDomain:

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_attribute('cell_type')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_iwalk(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_attribute('iwalk')
        return plt.gcf()

    def test_plot_attribute_bad_shape_1d(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        with pytest.raises(ValueError):
            _delta.show_attribute('free_surf_flag')

    def test_plot_attribute_bad_shape_3d(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        # not sure if there are any 3d arrays actually, so just make a fake one
        _delta.threedeearray = np.zeros((10, 10, 2))
        with pytest.raises(ValueError):
            _delta.show_attribute('threedeearray')

    def test_plot_attribute_missing_unknown(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        with pytest.raises(AttributeError):
            _delta.show_attribute('FAKE_NAME_ATTRIBUTE')

    def test_plot_attribute_numeric_bad_type(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        with pytest.raises(TypeError):
            _delta.show_attribute('dx')
        with pytest.raises(TypeError):
            _delta.show_attribute('CTR')

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_no_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_attribute('cell_type', grid=False)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_velocity(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_attribute('ux')
        return plt.gcf()

    def test_plot_domain_badtype(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        with pytest.raises(TypeError):
            _delta.show_attribute(42)

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_multiple_subplots(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(1, 3, figsize=(6, 4))
        _delta.show_attribute('eta', grid=False, ax=ax[1])
        _delta.show_attribute('cell_type', grid=False, ax=ax[0])
        _delta.show_attribute('ux', grid=False, ax=ax[2])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_withlabel(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        # This is a weak test, but it triggers coverage of the label lines.
        _delta.show_attribute('ux', label='')
        return plt.gcf()


class TestShowInd:

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_single_tuple(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind((30, 100))
        return plt.gcf()

    def test_plot_domain_cell_type_single_badtuple(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        with pytest.raises(ValueError):
            _delta.show_ind((100, 1100, 2100))

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_list_tuple(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind([(30, 100), (50, 100), (50, 150)])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_single_index(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind(2200)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_list_index(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind([1125, 1126, 1127, 1129, 2120])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_list_mix_tuple_index(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind([(15, 55), (15, 65), 1275, 1295, 2250])
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_multiple_index_calls(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind((3, 4))
        _delta.show_ind((3, 5))
        _delta.show_ind((3, 6))
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_multiple_diff_args(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind((3, 4), 'bs')
        _delta.show_ind((3, 5), 'r*')
        _delta.show_ind((3, 6), 'g^')
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_multiple_diff_kwargs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(_delta.cell_type, interpolation='none')
        ax.autoscale(False)
        _delta.show_ind((3, 4), marker='s', color='cyan')
        _delta.show_ind((3, 5), markersize=12)
        _delta.show_ind([(3, 4), (3, 5), (3, 6), 19, 20],
                        color='green', marker='.')
        return plt.gcf()


class TestCombinationsOfMultipleMethods:

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_domain_cell_type_no_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_attribute('cell_type', grid=False)
        _delta.show_ind([(3, 4), (3, 5), (3, 6), 19, 20],
                        color='green', marker='.')
        return plt.gcf()


class TestShowLine:

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_show_line_pts_Nx2_array(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        _arr = np.column_stack((np.arange(9), np.arange(11, 20)))
        assert _arr.shape[1] == 2

        fig, ax = plt.subplots(figsize=(5, 4))
        _delta.show_line(_arr, ax=ax)
        _delta.show_line(np.fliplr(_arr), 'r-.', ax=ax)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_show_line_set_points(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        _delta = DeltaModel(input_file=p)

        fig, ax = plt.subplots(figsize=(5, 4))
        np.random.seed(0)
        _delta.free_surf_walk_inds = np.tile(
            np.arange(100, 2900, step=_delta.eta.shape[1]),
            (12, 1))
        _shape = _delta.free_surf_walk_inds.shape
        _delta.free_surf_walk_inds += np.random.randint(-5, 5, size=_shape)

        _delta.show_line(
            _delta.free_surf_walk_inds.T, multiline=True, ax=ax)
        return plt.gcf()
