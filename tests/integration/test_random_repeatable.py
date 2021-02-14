from .. import utilities
import numpy as np

from pyDeltaRCM import DeltaModel


class TestModelIsReprodicible:

    def test_same_result_two_models(self, tmp_path):
        """Test consistency of two models initialized from same yaml."""
        p1 = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                      {'out_dir': tmp_path / 'out_dir_1',
                                       'seed': 10})
        p2 = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                      {'out_dir': tmp_path / 'out_dir_2',
                                       'seed': 10})

        # create and update first model
        ModelA = DeltaModel(input_file=p1)
        ModelA.update()
        ModelA.output_netcdf.close()
        # create and update second model
        ModelB = DeltaModel(input_file=p2)
        ModelB.update()
        ModelB.output_netcdf.close()

        # fields should be the same
        assert ModelA.time == ModelB.time
        assert ModelA._time_iter == ModelB._time_iter
        assert ModelA._save_iter == ModelB._save_iter
        assert ModelA._save_time_since_last == ModelB._save_time_since_last
        assert np.all(ModelA.uw == ModelB.uw)
        assert np.all(ModelA.ux == ModelB.ux)
        assert np.all(ModelA.uy == ModelB.uy)
        assert np.all(ModelA.depth == ModelB.depth)
        assert np.all(ModelA.stage == ModelB.stage)
        assert np.all(ModelA.strata_eta.todense() ==
                      ModelB.strata_eta.todense())
        assert np.all(ModelA.strata_sand_frac.todense() ==
                      ModelB.strata_sand_frac.todense())
