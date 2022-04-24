"""
Check that the dependency with InfraRender submodule works well
"""
import pytest


class TestInfraRender:

    def test_imports(self):
        try:
            from InfraRender import InfraRender
            from InfraRender.InfraRender import (AnalysisBySynthesis,
                                                 DispersionModelEstimator,
                                                 InverseRender)
            from InfraRender.util import MixedSpectra, EndmemberLibrary
            from InfraRender.util import (get_metrics,
                                          save_list_of_dicts,
                                          create_result_directory,
                                          consolidate_feely,
                                          create_experiment_directory)
        except ModuleNotFoundError as e:
            pytest.fail(
                ("Import functions from InfraRender submodule. "
                 "Are you at the project root?"), e)
