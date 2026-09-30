import unittest

from vimuq.uq.distribution import Bounds


class TestBounds(unittest.TestCase):
    def test__init_throws_error(self):
        lower = 1
        upper = lower - 0.1
        self.assertRaises(ValueError, Bounds, lower_bound=lower, upper_bound=upper)


if __name__ == "__main__":
    unittest.main()
