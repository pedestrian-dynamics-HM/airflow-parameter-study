import unittest

from vimuq.util.math import round_up_to_xth_place


class TestMath(unittest.TestCase):
    def setUp(self) -> None:
        self.numbers = [3, 73, 173, 8173]

    def test_round_up_to_xth_place_no_place_set(self):
        expected = [3, 80, 200, 9000]
        rounded = [round_up_to_xth_place(x) for x in self.numbers]
        self.assertListEqual(rounded, expected)

    def test_round_up_to_xth_place_place_1(self):
        expected = [10, 80, 180, 8180]
        rounded = [round_up_to_xth_place(x, place=1) for x in self.numbers]
        self.assertListEqual(rounded, expected)

    def test_round_up_to_xth_place_place_2(self):
        expected = [100, 100, 200, 8200]
        rounded = [round_up_to_xth_place(x, place=2) for x in self.numbers]
        self.assertListEqual(rounded, expected)


if __name__ == "__main__":
    unittest.main()
