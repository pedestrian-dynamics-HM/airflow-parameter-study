import unittest

from vimuq.uq.uq_experiment import Stage
from vimuq import _get_experiment_pkl_filename, get_experiment_dir_name


class TestVimuq(unittest.TestCase):
    def test__get_experiment_pkl_filename(self):
        pre = "prefix"
        self.assertEqual(
            _get_experiment_pkl_filename(Stage.PRE_RUN, prefix=pre),
            pre + "_pre_run.pkl",
        )

    def test__get_experiment_pkl_filename_suffix(self):
        pre = "prefix"
        suf = "suf"
        self.assertEqual(
            _get_experiment_pkl_filename(Stage.PRE_RUN, prefix=pre, suffix=suf),
            pre + "_pre_run" + f"-{suf}.pkl",
        )

    def test__get_experiment_pkl_filename_avrg(self):
        pre = "prefix"
        avrg = "avrgA"
        self.assertEqual(
            _get_experiment_pkl_filename(
                Stage.PRE_RUN, prefix=pre, avrg_technique=avrg
            ),
            pre + "_pre_run" + f"_{avrg}.pkl",
        )

    def test__get_experiment_pkl_filename_avrg_suf(self):
        pre = "prefix"
        suf = "suf"
        avrg = "avrgA"
        self.assertEqual(
            _get_experiment_pkl_filename(
                Stage.PRE_RUN, prefix=pre, avrg_technique=avrg, suffix=suf
            ),
            pre + "_pre_run" + f"_{avrg}" + f"-{suf}.pkl",
        )

    def test__get_experiment_dir_name(self):
        dir = "test"
        suffix = ["", None, "suf"]
        res = [get_experiment_dir_name(dir_name=dir, suffix=suf) for suf in suffix]
        self.assertEqual(res, [dir, dir, dir + "-" + suffix[2]])


if __name__ == "__main__":
    unittest.main()
