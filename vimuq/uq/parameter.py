"""model input parameter"""

from vimuq.uq.distribution import Distribution


class Parameter:
    def __init__(self, name: str, dist: Distribution):
        self.name = name
        self.dist = dist
