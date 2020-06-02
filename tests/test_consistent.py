# unit tests for consistent model outputs

import pytest

import sys
import os
import numpy as np

from utilities import test_DeltaModel

# need to create a simple case of pydeltarcm object to test these functions


def test_bed_after_one_update(test_DeltaModel):
    test_DeltaModel.update()
    # slice is: test_DeltaModel.eta[:5, 4]
    # print(test_DeltaModel.eta[:5, 4])

    _exp = np.array([-1., -0.840265, -0.9976036, -1., -1.])
    assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))
