import warnings
from typing import List


class PostProcessApplication:

    # valid functions corresponding to functions from pandas.core.GroupBy;
    # this set is not exhaustive
    # skipzero functionality is only kept for backward compatibility
    skipzero = "z_"
    VALID_FUNCS = {
        "sum",
        "min",
        "max",
        "mean",
        "median",
        "std",
        "var",
        "rank",
        "count",
    }
    VALID_OPERATORS = {">", "<", ">=", "<="}

    def __init__(self, qoi: str, function: str, indices: List[str] = list()):
        self.qoi = qoi

        if self.skipzero in function:
            warnings.warn(
                f"Skipping zeros with {self.skipzero} is deprecated."
                f"Consider using {function}_>_0 instead."
            )
            function = function.replace(self.skipzero, "") + "_>_0"

        self.qoi_suffix = function

        if "_" in function:
            function, operator, threshold = function.split("_")

            if operator in self.VALID_OPERATORS:
                self.operator = operator
                self.threshold = float(threshold)
        else:
            self.operator = None
            self.threshold = None

        if function not in self.VALID_FUNCS:
            raise ValueError(
                f"Argument function must be one of the following: "
                f"{self.VALID_FUNCS}"
            )
        else:
            self.function = function
        self.indices = indices

        # Index "id" must always remain as index in the data frame because it is
        # associated with the samples; put "id" as first index in the list
        self._add_index("id", 0)

    def _add_index(self, idx, position=-1):
        if idx in self.indices:
            self.indices.remove(idx)
        self.indices.insert(position, idx)

    def add_index_run_id(self):
        self._add_index("id", 0)
        self._add_index("run_id", 1)

    def get_new_qoi_name(self):
        return self.qoi + "_" + self.qoi_suffix
