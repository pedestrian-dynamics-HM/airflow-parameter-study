import math
import os
import warnings
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np

import pandas as pd
import scipy.stats as stats

from vimuq.experiments.plot.style import apply_style
from vimuq.uq.uq_experiment import UQExperiment

from itertools import combinations


class RestrictedDict(dict):
    _allowed_keys = set()
    _defaults = dict()

    def __init__(self, *, allowed_keys=None, **kwargs):

        try:
            iter(allowed_keys)
        except TypeError:
            print("'allowed_keys' not an iterable")
        else:
            self._allowed_keys = allowed_keys

        for key, value in kwargs.items():
            self.check_keys(key)
            self[key] = value

    def __setitem__(self, key, val):
        self.check_keys(key)
        dict.__setitem__(self, key, val)

    def check_keys(self, key):
        if key not in self._allowed_keys:
            raise KeyError(
                "key: {!r} not in allowed keys: {!r}".format(key, self._allowed_keys)
            )

    def set_defaults(self):
        for k, v in self._defaults.items():
            if k not in self.keys():
                self.__setitem__(key=k, val=v)


class SubplotDescriptions(RestrictedDict):
    _allowed_keys = {"title", "xlabel", "ylabel", "yticks", "ylim"}

    def __init__(self, *, allowed_keys=None, **kwargs):
        super().__init__(allowed_keys=self._allowed_keys, **kwargs)

    def copy(self):
        new = SubplotDescriptions()
        for k, v in self.items():
            new.__setitem__(k, v)
        return new

    def has_key(self, key):
        if key in self.keys():
            return True
        else:
            return False


class AdjustParams(RestrictedDict):
    _allowed_keys = {"left", "bottom", "right", "top", "wspace", "hspace"}

    def __init__(self, *, allowed_keys=None, **kwargs):
        super().__init__(allowed_keys=self._allowed_keys, **kwargs)


class LegendDict(RestrictedDict):
    _allowed_keys = {"title", "ncols", "entrydict", "location"}
    _defaults = {"title": None, "ncols": 1}

    def __init__(self, *, allowed_keys=None, **kwargs):
        super().__init__(allowed_keys=self._allowed_keys, **kwargs)
        self.set_defaults()


class MarkerDict(RestrictedDict):
    _allowed_keys = {"marker", "size", "filled", "edgewidth"}
    _defaults = {
        "marker": "o",
        "size": 2,
        "filled": True,
        "edgewidth": 1,  # matplotlib.rcParams["lines.markeredgewidth"]
        # does not work because rcParams are modified after defaults
        # are called
    }

    def __init__(self, *, allowed_keys=None, **kwargs):
        super().__init__(allowed_keys=self._allowed_keys, **kwargs)
        self.set_defaults()


class BasicPlot:
    _FIG_DIR = "figures"

    def __init__(self):
        self.apply_vimuq_rcparams()
        self.figure = None
        self.axes = None

    def save(
        self, parent_dir: str, name: str = "", file_extension: str = "pdf", dpi="figure"
    ):
        path2fig = os.path.join(parent_dir, self._FIG_DIR)
        if not os.path.exists(path2fig):
            os.makedirs(path2fig)
        file_name = name + "." + file_extension
        path2file = os.path.join(path2fig, file_name)
        self.figure.savefig(path2file, dpi=dpi)

    def apply_vimuq_rcparams(self, style="vimuq1", use_case=""):
        apply_style(sty=style, use_case=use_case)


class SamplePlot(BasicPlot):
    def __init__(self, experiment: UQExperiment, label_map, n_cols=2):
        super().__init__()
        self.experiment = experiment
        self.label_map = label_map
        self.n_cols = n_cols

    def make(
        self,
        figsize=None,
        marker_size=36,
        type="hist",
    ):
        samples = pd.DataFrame(self.experiment.parameter_variations)
        bounds = [
            np.array([p.dist.lower_bound, p.dist.upper_bound])
            for p in self.experiment.model.parameter_space.parameters
        ]
        ticks = [np.linspace(start=b[0], stop=b[1], num=3) for b in bounds]

        # rename parameters for plotting
        samples.rename(mapper=self.label_map, axis="columns", inplace=True)

        # create figure
        dim = samples.shape[1]
        n_plots = dim + 1
        n_rows = int(math.ceil(n_plots / self.n_cols))
        self.figure, self.axes = plt.subplots(
            nrows=n_rows, ncols=self.n_cols, figsize=figsize
        )
        # add histogram for each dimension
        if type == "hist":
            for ax, col, bound, tick in zip(
                self.axes.flatten()[:dim], samples.columns, bounds, ticks
            ):
                ax.hist(x=samples[col], range=bound)
                ax.set_xlabel(col)
                ax.set_xlim(bound)
                ax.set_xticks(tick)

        cols_combis = list(combinations(samples.columns, 2))
        bound_combis = list(combinations(bounds, 2))
        ticks_combis = list(combinations(ticks, 2))
        # add scatter plots for all 2D combinations of the parameter space
        if type == "scatter":
            for ax, col2, bound2, tick2 in zip(
                self.axes.flatten()[:dim], cols_combis, bound_combis, ticks_combis
            ):
                ax.scatter(x=samples[col2[0]], y=samples[col2[1]], s=marker_size)
                ax.set_xlabel(col2[0])
                ax.set_xlim(bound2[0])
                ax.set_xticks(tick2[0])
                ax.set_ylabel(col2[1])
                ax.set_ylim(bound2[1])
                ax.set_yticks(tick2[1])

        # add 3d scatter plot for three dimensions dims defined in the order x, y, z
        dims = [0, 1, 2]
        self.axes.flatten()[dim].remove()
        ax = self.figure.add_subplot(n_rows, self.n_cols, n_plots, projection="3d")
        ax.scatter(
            xs=samples.iloc[:, dims[0]],
            ys=samples.iloc[:, dims[1]],
            zs=samples.iloc[:, dims[2]],
            s=marker_size,
        )
        ax.set_xlabel(samples.columns[dims[0]])
        ax.set_xlim(bounds[dims[0]])
        ax.set_xticks(ticks[dims[0]])
        ax.set_ylabel(samples.columns[dims[1]])
        ax.set_ylim(bounds[dims[1]])
        ax.set_yticks(ticks[dims[1]])
        ax.set_zlabel(samples.columns[dims[2]])
        ax.set_zlim(bounds[dims[2]])
        ax.set_zticks(ticks[dims[2]])


class ForwardUQPlot(BasicPlot):
    def __init__(self, df: pd.DataFrame, axis_properties: dict, n_cols=2):
        super().__init__()
        self.df = df.copy()
        self.ax_props = axis_properties
        self.qois = axis_properties.keys()
        self.n_cols = n_cols
        self.n_plots = len(axis_properties.keys())
        self.n_rows = int(math.ceil(self.n_plots / n_cols))

    def make_plot(
        self,
        type,
        figsize=None,
        fig_rel_height=1,
        hist_density=False,
        cumulative=False,
        skip_qoi=[],
    ):
        if fig_rel_height == 1:
            fs = figsize
        else:
            fs = [
                matplotlib.rcParams["figure.figsize"][0],
                matplotlib.rcParams["figure.figsize"][1] * fig_rel_height,
            ]
        if self.figure is None and self.axes is None:
            self.figure, self.axes = plt.subplots(
                nrows=self.n_rows,
                ncols=self.n_cols,
                figsize=fs,
            )

        if type == "line":
            # filter values by qoi
            self.df = self.df[self.df.index.isin(self.qois, level="qoi")]

            # sort values by ax_props.keys()
            order = "order"
            self.df.insert(
                loc=0, column=order, value=self.df.index.get_level_values("qoi")
            )
            self.df[order] = self.df[order].map(
                dict(zip(self.qois, np.arange(len(self.qois))))
            )
            self.df.sort_values(by=order, inplace=True)

            by_qoi = self.df.groupby(["qoi"])
            target_subplots = zip(by_qoi.groups.keys(), self.axes.flatten())
            for name, ax in target_subplots:
                group = by_qoi.get_group(name).copy()  # take only a copy of the
                # desired group to allow for altering the groupby object, e.g.,
                # sorting
                props = self.ax_props[name]
                if name not in skip_qoi:
                    ForwardUQPlot.make_line_subplot(ax, group, props)

        elif type in ["hist", "kde"]:
            target_subplots = zip(self.axes.flatten(), self.ax_props.keys())
            for ax, col in target_subplots:
                data = self.df[col]
                props = self.ax_props[col]
                if col not in skip_qoi:
                    if type == "hist":
                        ForwardUQPlot.make_hist_subplot(
                            data=data,
                            ax=ax,
                            props=props,
                            density=hist_density,
                            cumulative=cumulative,
                        )
                    elif type == "kde":
                        ForwardUQPlot.make_kde_subplot(
                            data=data, ax=ax, props=props, cumulative=cumulative
                        )

    @staticmethod
    def make_line_subplot(ax, group, props, color=None, fill_alpha=0.1):
        if "sort_x_by" in props.keys():
            group.sort_values(by=props["sort_x_by"], inplace=True)
        if props["x"] in group.index.names:
            x = group.index.get_level_values(props["x"]).tolist()
        else:
            x = group[props["x"]]
        y = group[props["y"]].values
        p1 = ax.plot(x, y, color=color, label=props["legend"])
        if "y_fill" in props.keys():
            ax.fill_between(
                x=x,
                y1=group[props["y_fill"][0]],
                y2=group[props["y_fill"][1]],
                alpha=fill_alpha,
                color=color,
            )

        # TODO this does not work properly
        #
        #   # p2 is a dummy just to create the legend patch for the
        #   # content created by fill_between
        #   p2 = ax.fill(np.NaN, np.NaN, color=color, alpha=fill_alpha)
        #
        # # p1 and p2 are required for combined legend (line and fill_between)
        # if "legend" in props.keys():
        #     h, l = ax.get_legend_handles_labels()
        #     ax.legend(h + [(p2[0], p1[0])], l + [props["legend"]])

        if "legend" in props.keys():
            ax.legend()

        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="x_axis")
        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="y_axis")

    @staticmethod
    def make_hist_subplot(
        data, ax, props, density=False, color=None, alpha=None, cumulative=False
    ):
        legend = None
        if "legend" in props.keys():
            legend = props["legend"]

        bins = None
        if "bins" in props.keys():
            bins = props["bins"]
            if "x_axis" in props.keys():
                if "lim" in props["x_axis"].keys():
                    bins = np.linspace(
                        start=props["x_axis"]["lim"][0],
                        stop=props["x_axis"]["lim"][1],
                        num=props["bins"],
                        endpoint=True,
                    )

        ax.hist(
            data,
            bins=bins,
            density=density,
            label=legend,
            color=color,
            alpha=alpha,
            cumulative=cumulative,
        )

        if "legend" in props.keys():
            ax.legend()
        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="x_axis")
        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="y_axis")

    @staticmethod
    def make_kde_subplot(data, ax, props, x_num=100, color=None, cumulative=False):
        kernel = stats.gaussian_kde(dataset=data)

        # get x points from xlim
        x_lower = data.min()
        x_upper = data.max()
        if "x_axis" in props.keys():
            if "lim" in props["x_axis"].keys():
                x_lower = props["x_axis"]["lim"][0]
                x_upper = props["x_axis"]["lim"][1]
        x = np.linspace(x_lower, x_upper, num=x_num)

        # estimate y data with kde
        y_kde_estimate = kernel.evaluate(x)

        # calculate cumulative kde / cdf
        if cumulative:
            y_kde_estimate = y_kde_estimate.cumsum()

        legend = None
        if "legend" in props.keys():
            legend = props["legend"] + " (KDE)"
        ax.plot(x, y_kde_estimate, label=legend, color=color)

        if "legend" in props.keys():
            ax.legend()

        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="x_axis")
        ForwardUQPlot.set_axis_properties(ax=ax, properties=props, axis="y_axis")

    @staticmethod
    def set_axis_properties(ax, properties, axis):
        x_axis = "x_axis"
        y_axis = "y_axis"

        if axis in properties.keys():
            p = properties[axis]
            if axis == x_axis:
                if "label" in p.keys():
                    ax.set_xlabel(p["label"])
                if "lim" in p.keys():
                    ax.set_xlim(p["lim"])
                if "x_scale" in p.keys():
                    ax.set_xscale(p["x_scale"])
            elif axis == y_axis:
                if "label" in p.keys():
                    ax.set_ylabel(p["label"])
                if "lim" in p.keys():
                    ax.set_ylim(p["lim"])
                if "tick_format" in p.keys():
                    ax.ticklabel_format(
                        axis="y",
                        style=p["tick_format"][0],
                        scilimits=p["tick_format"][1],
                    )
        else:
            warnings.warn(
                f"Cannot handle specified axis '{axis}'. "
                f"Use either '{x_axis}' or '{y_axis}'"
            )


class SensitivityPlot:
    """
    Defines how to plot Sobol' sensitivity indices. This can be used either as
    single or as one of multiple subplots in a Plot. Plot provides the data set
    as a dataframe with multi index. The index values and index names mentioned
    here refer to this dataframe.

    Args:
        main_value          The index value defining a subset of data to be
                            considered. Typically, this is the name of the
                            quantity of interest, e.g. "pedIdKey_degExposure".

        idx_main            The index name that corresponds to main_value.
                            Typically, the index is the quantity of interest
                            ("qoi").

        type                Type of plot: see _allowed_types

        descriptions        SubplotDescriptions; defines also ylim of this plot

        idx_xticks          The index name defining how the data is grouped for
                            each major xtick. Typically uncertain parameter
                            ("param") or interacting parameters ("interaction").

        data_series_value   Select only specific values among all possible /
                            available values

        idx_data_series     The index name defining how the data is grouped for
                            each data series (e.g. for each bar/line). For
                            example, if the results of multiple experiments are
                            plotted within one SensitivityPlot, one can define
                            one data series (bar/line) for each experiment.

                            If type=="bar", plots each data series as separate
                            bars next to each other.

                            If type=="line" or "points", plots each data series
                            on top.

                            Gets labels assigned with same transparency (alpha).

                            Typical values: "experiment", "avrg", "n_samples"

        idx_alpha           This is deprecated. Use colors_data_series instead.

                            The index name defining how the data is grouped for
                            different alpha (transparency) values.
                            Typical value: "run_id"

        colors_data_series  Defines the colors for each data series.

        idx_minor_x         The index name defining the minor x-ticks.
                            Typical value if sensitivity is plotted for each
                            pedestrian: "pedestrianId".

        show_xticks         Defines whether x-ticks are shown for none, major,
                            minor x-tick values.

        shaded_xticks       Defines whether major x-ticks have a shaded
                            background. Either bool or float defining the alpha/
                            transparency.

        x_ticks_width       Defines the width of (shaded) column around major
                            x-ticks.

        data_col            Column name defining which column is plotted.
                            Typical values: "ST", "S2"

        legend              Defines a legend; the location inside the
                            sensitivity plot can be defined through matplotlib rcParams

        marker              Defines appearance of markers in the plot.
                            It type=="points", defines the marker for each data
                            point.
                            If type=="line", adds markers to line plots.
                            Not compatible with bar charts.

        fill_minor_x_vals   Allows to manually insert to the dataset that is
                            plotted along minor x-ticks (e.g. an additional
                            pedestrian ID). The corresponding y value is set to
                            nan.

        plot_conf_interval  If True, plots error bars (for bar charts) or shaded
                            intervals (for line charts). Uses column
                            <data_col>"_conf"
    """

    # ToDo create child classes of SensitivityPlot for each _allowed_types
    _allowed_types = {"bar", "line", "points"}

    def __init__(
        self,
        main_value,
        idx_main="qoi",
        type: str = "bar",
        descriptions: SubplotDescriptions = None,
        idx_xticks: str = "param",
        idx_data_series: str = None,
        data_series_value_selection: List = None,
        idx_alpha: str = None,
        colors_data_series=matplotlib.cm.get_cmap("tab10").colors,
        idx_minor_x: str = None,
        show_xticks: str = None,
        shaded_xticks: Union[float, bool] = True,
        xticks_width: float = 0.85,
        data_col: str = "ST",
        legend: LegendDict = None,
        marker: Union[list, MarkerDict] = None,
        hatches: Union[list, None] = None,
        fill_minor_x_vals: List[float] = None,
        plot_conf_interval: bool = False,
    ):
        self.main_value = main_value
        self.idx_main = idx_main

        if type not in self._allowed_types:
            raise ValueError(f"Argument 'type' must be {self._allowed_types}.")
        else:
            self.type = type
        if type in ["line", "points"] and not idx_minor_x:
            raise ValueError(
                f"Arugment 'idx_minor_x' must be specified if 'type' == 'line' or 'type' == 'points'."
            )

        if descriptions is None:
            self.descriptions = SubplotDescriptions()
        else:
            self.descriptions = descriptions

        self.idx_minor_x = idx_minor_x
        self.show_xticks = show_xticks

        self.marker = marker
        if type == "points" and self.marker is None:
            self.marker = MarkerDict()

        self.hatches = hatches

        self.fill_minor_x_vals = fill_minor_x_vals

        self.idx_xticks = idx_xticks
        self.ds_values = data_series_value_selection
        self.idx_data_series = idx_data_series
        self.colors_data_series = colors_data_series
        self.idx_alpha = idx_alpha

        self.shaded_xticks = shaded_xticks
        self.xticks_width = xticks_width
        self.data_col = data_col

        self.legend = legend

        self.conf = plot_conf_interval

    def subplot_from_data(self, ax: matplotlib.axes.Axes, data_frame: pd.DataFrame):
        data_s = PlotDataFrame.select_by_index_or_column_values(
            df=data_frame,
            level=self.idx_main,
            select_values=[self.main_value],
        )

        x_ticks_values = data_s.index.get_level_values(
            level=self.idx_xticks
        ).drop_duplicates()

        delta = 1
        x_majors = np.arange(len(x_ticks_values), step=delta)
        minor_x_ticks = np.empty(0)

        legend_hdl = []

        for x_val, x_major in zip(x_ticks_values, x_majors):

            if self.shaded_xticks:
                if isinstance(self.shaded_xticks, float):
                    a = self.shaded_xticks
                else:
                    a = 0.05
                x_major_lower = x_major - delta * self.xticks_width / 2
                x_major_upper = x_major + delta * self.xticks_width / 2
                if "ylim" in self.descriptions.keys():
                    shaded_ylim = self.descriptions["ylim"]
                else:
                    shaded_ylim = [0, 1]
                ax.fill_betweenx(
                    y=shaded_ylim,
                    x1=x_major_lower,
                    x2=x_major_upper,
                    facecolor="k",
                    edgecolor="none",
                    alpha=a,
                )

            data_x = PlotDataFrame.select_by_index_or_column_values(
                df=data_s, level=self.idx_xticks, select_values=[x_val]
            )

            if self.idx_data_series:
                level_ds = self.idx_data_series
            else:
                level_ds = self.idx_xticks

            data_series_values = data_x.index.get_level_values(
                level=level_ds
            ).drop_duplicates()

            if self.ds_values:
                data_series_values = data_series_values[
                    data_series_values.isin(self.ds_values)
                ]
            n_bars = len(data_series_values)
            bar_width = self.xticks_width / n_bars
            if n_bars == 1:
                x_minors_bar = np.zeros(1)
            else:
                x_minors_bar = np.linspace(
                    start=(-self.xticks_width + bar_width) / 2,
                    stop=(self.xticks_width - bar_width) / 2,
                    endpoint=True,
                    num=n_bars,
                )

            # define markers
            if not isinstance(self.marker, list):
                markers = [self.marker] * len(data_series_values)
            else:
                markers = self.marker

            # define hatches
            if not isinstance(self.hatches, list):
                hatches = [self.hatches] * len(data_series_values)
            else:
                hatches = self.hatches

            for ds_val, color, x_minor_b, marker, hatch in zip(
                data_series_values,
                self.colors_data_series,
                x_minors_bar,
                markers,
                hatches,
            ):

                data_ds = PlotDataFrame.select_by_index_or_column_values(
                    df=data_x, level=level_ds, select_values=[ds_val]
                )

                if self.idx_alpha:
                    level_alpha = self.idx_alpha
                else:
                    level_alpha = level_ds
                alpha_values = data_ds.index.get_level_values(
                    level=level_alpha
                ).drop_duplicates()
                for a_val in alpha_values:
                    data_a = PlotDataFrame.select_by_index_or_column_values(
                        df=data_ds, level=level_alpha, select_values=[a_val]
                    )

                    if len(alpha_values) >= 1:
                        alpha = 1
                    else:
                        alpha = 0.25

                    if self.type == "bar":

                        x = x_major + x_minor_b
                        y = data_a[self.data_col].copy()
                        if self.conf:
                            yerr = data_a[self.data_col + "_conf"]
                            error_kw = {"capsize": 1, "linewidth": 0.75}
                        else:
                            yerr = None
                            error_kw = dict()

                        if hatch:
                            bar_color = "none"
                            bar_edge_color = color
                        else:
                            bar_color = color
                            bar_edge_color = color

                        ax.bar(
                            x=x,
                            height=y,
                            width=bar_width,
                            yerr=yerr,
                            error_kw=error_kw,
                            hatch=hatch,
                            color=bar_color,
                            edgecolor=bar_edge_color,
                            lw=0.05,
                            alpha=alpha,
                        )

                    elif self.type in ["line", "points"]:
                        # define x data
                        x_minor_l = (
                            data_a.index.get_level_values(self.idx_minor_x)
                            .copy()
                            .__array__()
                        )
                        y = data_a[self.data_col].copy().__array__()
                        yerr = data_a[self.data_col + "_conf"].copy().__array__()

                        # fill missing x values and corresponding y values
                        if self.fill_minor_x_vals:
                            for v in self.fill_minor_x_vals:
                                if v not in x_minor_l:
                                    x_minor_l = np.append(
                                        arr=x_minor_l, values=self.fill_minor_x_vals
                                    )
                                    x_minor_l = np.sort(x_minor_l)
                                else:
                                    raise ValueError(
                                        f"Value {v} in fill_minor_x_vals already present in data set. Cannot be filled."
                                    )
                            for v in self.fill_minor_x_vals:
                                index = np.where(x_minor_l == v)[0][0]
                                y = np.insert(arr=y, obj=index, values=np.nan)
                                yerr = np.insert(arr=yerr, obj=index, values=np.nan)

                        # normalize to values between 0, 1
                        x_minor_l = (x_minor_l - x_minor_l.min()) / (
                            x_minor_l - x_minor_l.min()
                        ).max()
                        # scale to values between -+xticks_width/2
                        x_minor_l = (x_minor_l - 0.5) * self.xticks_width
                        # shift by x_major
                        x = x_minor_l + x_major

                        # collect minor x ticks
                        if minor_x_ticks.any():
                            minor_x_ticks = np.concatenate([minor_x_ticks, x])
                        else:
                            minor_x_ticks = x

                        if self.type == "line":
                            if marker is not None:

                                edge = "none"
                                face = color
                                if "filled" in marker.keys():
                                    if not marker["filled"]:
                                        edge = color
                                        face = "none"

                                ax.plot(
                                    x,
                                    y,
                                    color=color,
                                    lw=0.5,
                                    alpha=alpha,
                                    marker=marker["marker"],
                                    markersize=marker["size"],
                                    markeredgewidth=marker["edgewidth"],
                                    markerfacecolor=face,
                                    markeredgecolor=edge,
                                )
                            else:
                                ax.plot(x, y, color=color, lw=0.5, alpha=alpha)
                            if self.conf:
                                y1 = y + yerr
                                y2 = y - yerr

                                # define different alphas for face and edgecolor
                                alpha_conf = 0.25
                                facecolor = list(color)
                                facecolor.append(alpha_conf)
                                edgecolor = list(color)
                                edgecolor.append(alpha_conf + 0.05)

                                ax.fill_between(
                                    x=x,
                                    y1=y1,
                                    y2=y2,
                                    facecolor=tuple(facecolor),
                                    edgecolor=tuple(edgecolor),
                                    lw=0.1,
                                )
                        elif self.type == "points":
                            ax.plot(x, y, linestyle="none", marker=marker)

                    if self.legend and len(legend_hdl) < n_bars and self.type == "bar":
                        legend_hdl.append(
                            mpatches.Patch(
                                color=color, label=self.legend["entrydict"][ds_val]
                            )
                        )

        ax.set_xticks(x_majors)
        ax.set_xticklabels(x_ticks_values)
        if not self.show_xticks == "major":
            # hide major x-ticks by setting length to 0
            ax.tick_params(which="major", axis="x", color="k", length=0)

        if self.show_xticks == "minor":
            ax.set_xticks(minor_x_ticks, minor=True)
            which = "both"
            for x in x_majors:
                if x not in minor_x_ticks:
                    which = "minor"
            ax.tick_params(
                which=which,
                axis="x",
                direction="in",
                width=0.75,
                length=1.5,
                color="k",
            )

        keys = self.descriptions.keys()
        if "xlabel" in keys:
            ax.set_xlabel(self.descriptions["xlabel"])
        if "yticks" in keys:
            ax.set_yticks(self.descriptions["yticks"])
        if "ylim" in keys:
            ax.set_ylim(self.descriptions["ylim"])
        if "ylabel" in keys:
            ax.set_ylabel(self.descriptions["ylabel"])
        if "title" in keys:
            ax.set_title(self.descriptions["title"])

        if self.legend:
            if "location" in self.legend.keys():
                loc = self.legend["location"]
            else:
                loc = None
            if self.legend["title"]:
                title = self.legend["title"]
            else:
                title = None
            if legend_hdl:
                ax.legend(
                    handles=legend_hdl, title=title, ncol=self.legend["ncols"], loc=loc
                )
            else:
                labels = self.legend["entrydict"].values()
                lines = ax.get_lines()[: len(labels)]
                dummy_lines = []
                for l in lines:
                    face = l.get_markerfacecolor()
                    edge = l.get_color()
                    dummy_lines.append(
                        Line2D(
                            [],
                            [],
                            color=l.get_color(),
                            marker=l.get_marker(),
                            markeredgewidth=0.5,
                            markerfacecolor=face,
                            markeredgecolor=edge,
                            markersize=l.get_markersize() * 8,
                            lw=l.get_linewidth(),
                        )
                    )
                ax.legend(dummy_lines, labels, loc=loc)


class Plot(BasicPlot):
    def __init__(
        self,
        data: pd.DataFrame,
        title: str = None,
        rowwise_subplots: bool = True,
        n_subplot_cols: int = 1,
        tight_layout: bool = True,
    ):
        super().__init__()
        self.data = data
        self.title = title
        self.rowwise_subplots = rowwise_subplots
        self.n_subplot_cols = n_subplot_cols
        self.tight_layout = tight_layout

        self.subplots = dict()
        self.n_subplots: int = 0
        self.n_subplot_rows: int = 1

    def add_subplot(self, subplot: SensitivityPlot, loc: int = None):
        if not loc and self.n_subplots == 0:
            loc = 0
        elif not loc:
            loc = max(self.subplots.keys()) + 1
        self.subplots.update({loc: subplot})
        self.n_subplots += 1
        self.n_subplot_rows = math.ceil(self.n_subplots / self.n_subplot_cols)

    def adjust_margins_for_big_legend(self):
        height = self.figure.get_figheight() * 1.2
        width = self.figure.get_figwidth()
        self.figure.set_size_inches((width, height))

        self.figure.subplots_adjust(
            bottom=0.15, top=0.92, wspace=0.25, hspace=0.65, right=0.98, left=0.06
        )

    def shift_xlabels_to_top(self, subplot_map: dict):
        axes = self.figure.get_axes()
        for k, ax in enumerate(axes):
            if k in subplot_map.keys():
                xlabel = subplot_map[k]["xlabel"]
                minor_x = subplot_map[k]["show_minor_x"]
            else:
                xlabel = ""
                minor_x = False

            # move major x-ticks above subplot
            xticks = ax.get_xticklabels()
            for xtick in xticks:
                ax.text(
                    x=xtick.get_position()[0],
                    y=ax.get_ylim()[1] * 1.02,
                    s=xtick.get_text(),
                    ha="center",
                    va="bottom",
                )
            ax.set_xticklabels([])

            title = ax.get_title()
            # add minor x-tick labels
            if minor_x:
                n_data_series = subplot_map[k]["n_data_series"]
                n_xticks = subplot_map[k]["n_xticks"]
                x_minorticks = ax.axes.xaxis.get_minorticklabels()
                x_majorticks = ax.axes.xaxis.get_majorticklabels()
                x = []
                for i in range(len(x_majorticks)):
                    i_ = i * n_data_series * (n_xticks - 1)
                    x1 = x_minorticks[i_ : i_ + (n_xticks - 1)]
                    x1a = x1[0 : (n_xticks - 1)]
                    x1b = x1[(n_xticks - 1) :]
                    x1a.append(x_majorticks[i])
                    x1a.sort(key=lambda l: l.get_position()[0])
                    # add labels
                    x1a[-n_xticks].set_text("$1$")
                    x1a[-1].set_text(f"${str(n_xticks)}$")
                    x += x1a + x1b
                ax.axes.xaxis.set_ticks(
                    ticks=[xx.get_position()[0] for xx in x],
                    labels=[xx.get_text() for xx in x],
                    minor=True,
                )

            # move title up
            ax.set_title(label=title, y=1.1)

            # add x label
            if xlabel != "":
                ax.set_xlabel(xlabel)

    def make_plot(self, figsize=None):
        self.figure, self.axes = plt.subplots(
            nrows=self.n_subplot_rows,
            ncols=self.n_subplot_cols,
            figsize=figsize,
        )

        if self.title != "":
            self.figure.suptitle(self.title)

        s_row = 0
        s_col = 0

        for s_idx, subplot in zip(range(self.n_subplots), self.subplots.values()):
            ax = self._get_axes(row=s_row, col=s_col)
            # hide axes if no subplot expected / avoid empty axes
            if s_idx < self.n_subplots:
                subplot.subplot_from_data(ax, self.data)
            else:
                ax.axis("off")

            # ToDo: refactoring - replace this by iteration over flattend axes
            s_row, s_col = self._get_next_row_col_pair(row=s_row, col=s_col)

        if self.tight_layout:
            self.figure.tight_layout()

        if self.title:
            self.figure.suptitle(self.title)

    def _get_next_row_col_pair(self, row, col):
        if self.rowwise_subplots:
            col += 1
            if col == self.n_subplot_cols:
                col = 0
                row += 1
        else:
            row += 1
            if row == self.n_subplot_rows:
                row = 0
                col += 1

        return row, col

    def _get_axes(self, row: int, col: int):
        if self.n_subplot_rows + self.n_subplot_cols == 2:
            ax_ = self.axes
        elif self.n_subplot_rows <= 1 or self.n_subplot_cols <= 1:
            ax_ = self.axes[row + col]
        else:
            ax_ = self.axes[row, col]
        return ax_

    def get_fig(self):
        return self.figure


class PlotDataFrame(pd.DataFrame):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_index(df: pd.DataFrame, index_name: str, value):
        df_ = df.copy()
        df_.insert(loc=0, column=index_name, value=value)
        df_.set_index(keys=index_name, append=True, drop=True, inplace=True)
        return df_

    @staticmethod
    def remove_none_index_levels(df: pd.DataFrame):
        df_ = df.copy()
        idx_names = df_.index.names.copy()
        while None in idx_names:
            none_level = idx_names.index(None)
            df_ = df_.droplevel(level=none_level)
            idx_names = df_.index.names.copy()
        return df_

    @staticmethod
    def concat_unequal_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
        """Harmonizes dataframes and concatenates them. df1 is the base dataframe
        into which df2 is integrated."""

        if df1.empty:
            df_1 = df2.copy()
        else:
            df_1, df_2 = PlotDataFrame.harmonize_indices(df1.copy(), df2.copy())
            df_1 = pd.concat([df_1.copy(), df_2.copy()])

        return df_1

    @staticmethod
    def harmonize_indices(df1: pd.DataFrame, df2: pd.DataFrame, value=np.nan):
        """Brings indices of df1 and df2 in alignment. That is, adds indices from
        df2 that are not in df1 to df1 and vice versa. The added index values are
        defined by value.
        The indices are sorted by the final order of the indices of df1.
        """
        df_1 = df1.copy()
        df_2 = df2.copy()

        idx1 = set(df_1.index.names.copy())
        idx2 = set(df_2.index.names.copy())

        idx1_to_2 = [x for x in idx1 if x not in idx2]
        idx2_to_1 = [x for x in idx2 if x not in idx1]

        for idx in idx2_to_1:
            df_1 = PlotDataFrame.add_index(df_1, idx, value=value)
        for idx in idx1_to_2:
            df_2 = PlotDataFrame.add_index(df_2, idx, value=value)

        idx1 = df_1.index.names.copy()
        df_2 = df_2.reorder_levels(order=idx1)
        idx2 = df_2.index.names.copy()

        # This error should only occur if a dataframe has multiple indices with the
        # same name. In this case, the dataframe should be corrected.
        if idx1 != idx2:
            raise UserWarning(
                f"Indices are not equal {idx1} and {idx2}. Check dataframes for duplicated indices."
            )

        return df_1, df_2

    @staticmethod
    def select_by_index_or_column_values(
        df: pd.DataFrame, level: str, select_values: list
    ):
        df_ = df.copy()

        if level in df_.columns.to_list():
            df_ = df_[df_[level].isin(select_values)]
        elif level in df_.index.names.copy():
            df_.reset_index(level=level, inplace=True)
            df_ = df_[df_[level].isin(select_values)]
            df_.set_index(keys=level, append=True, drop=True, inplace=True)
        else:
            raise UserWarning(
                f"Column or index {level} not found in dataframe\n\n{df_.head()}."
            )

        return df_

    @staticmethod
    def uq_output_dict_to_df(
        uq_output_dict: dict, uq_output_df: pd.DataFrame = pd.DataFrame()
    ):
        # convert dict of dataframes to one dataframe
        for qoi, qoi_df in uq_output_dict.items():
            qoi_df = PlotDataFrame.add_index(df=qoi_df, index_name="qoi", value=qoi)
            qoi_df = PlotDataFrame.remove_none_index_levels(qoi_df)
            uq_output_df = PlotDataFrame.concat_unequal_dataframes(uq_output_df, qoi_df)

        return uq_output_df

    @staticmethod
    def transform_to_second_order_df(df: pd.DataFrame, rename_params=None):
        # drop all columns that do not correspond to second order indices
        df.drop(
            labels=["S1", "S1_conf", "ST", "ST_conf"],
            axis="columns",
            inplace=True,
            errors="ignore",
        )
        # now df.column.names should contain only tuples in which the first
        # element is either "S2" or "S2_conf"
        allowed_keys = ["S2", "S2_conf"]

        # rename columns
        col_names = df.columns.to_list()
        mapper = dict()
        for col_name in col_names:
            if type(col_name) is not tuple:
                raise ValueError(
                    f"Column name '{col_name}' is not of expected type tuple."
                )

            tuple_obj_1 = col_name[0]
            tuple_obj_2 = col_name[1]
            if tuple_obj_1 not in allowed_keys:
                raise ValueError(
                    f"Column name '{col_name}'[0] is one of expected keys: {allowed_keys}."
                )

            conf_key = "_conf"
            if conf_key not in tuple_obj_1:
                conf_key = ""
            if rename_params:
                new_name = rename_params[tuple_obj_2]
            else:
                new_name = tuple_obj_2
            new_name = new_name + conf_key

            mapper.update({col_name: new_name})

        df.rename(mapper=mapper, axis="columns", inplace=True)

        # reshape dataframe and obtain
        # multiindex: <old indices>/second order param, columns: S2, S2_conf
        df_ = pd.DataFrame()
        df_conf = pd.DataFrame()
        for col in df.columns.to_list():
            if "_conf" in col:
                val_name = "S2_conf"
                df_tmp = pd.melt(
                    df,
                    value_vars=[col],
                    var_name="param2",
                    value_name=val_name,
                    ignore_index=False,
                )
                df_tmp["param2"] = df_tmp.param2.str.replace("_conf?", "", regex=True)
                if df_conf.empty:
                    df_conf = df_tmp
                else:
                    df_conf = pd.concat([df_conf, df_tmp])
            else:
                val_name = "S2"
                df_tmp = pd.melt(
                    df,
                    value_vars=[col],
                    var_name="param2",
                    value_name=val_name,
                    ignore_index=False,
                )
                if df_.empty:
                    df_ = df_tmp
                else:
                    df_ = pd.concat([df_, df_tmp])
        df_.set_index("param2", append=True, drop=True, inplace=True)
        df_conf.set_index("param2", append=True, drop=True, inplace=True)

        df_ = pd.concat([df_, df_conf], axis="columns")

        # create index for interaction of param and param2
        df_.reset_index(level=["param", "param2"], drop=False, inplace=True)
        df_["interaction"] = df_["param"] + "_" + df_["param2"]
        df_.set_index("interaction", append=True, drop=True, inplace=True)
        df_.drop(labels=["param", "param2"], axis="columns", inplace=True)
        return df_

    @staticmethod
    def extract_second_order_df(
        df: pd.DataFrame,
        param_abbr: dict,
        drop_interact: Union[bool, List[str]] = True,
        interact_abbr: dict = None,
    ):
        """
        df              dataframe
        param_abbr      dict that defines how second order parameters should be
                        renamed. For example:
                        param_abbr = {
                            "param1_old_name": "p1",
                            "param2_old_name": "p2", ...
                        }
        drop_interact   bool: if True, drops all rows where all entries are nan.
                        Attention! This also drops nan values that are actually
                        required in the plot.
                        list: of parameter interactions that should not be
                        considered in the plot. The parameter interactions use
                        the new names for the parameters as defined by
                        param_abbr, and they are linked by an underscore (see
                        PlotDataFrame.transform_to_second_order_df).
                        For example:
                        drop_interact =
                        ["p1_p1", "p2_p2", ..., "p2_p1", "p3_p1", "p3_p2", ...]
        interact_abbr   dict that defines how parameter interactions should be
                        renamed. (This is how the interactions later appear in
                        the plot.) For example:
                        interact_abbr = {
                            "p1_p2": "$S_{2}(p1, p2)$",
                            "p1_p3": "$S_{2}(p1, p3)$", ...
                        }
        """

        # get dataframe containing only second order sensitivity indices
        df_s2 = df.copy()
        df_s2 = PlotDataFrame.transform_to_second_order_df(
            df_s2, rename_params=param_abbr
        )

        # drop redundant interactions
        if isinstance(drop_interact, bool):
            if drop_interact:
                df_s2.dropna(axis=0, how="all", inplace=True)
                warnings.warn(
                    "Dropping nan values. This may result in distorted visualization."
                )
        if isinstance(drop_interact, List):
            df_s2.drop(
                drop_interact,
                level="interaction",
                axis="index",
                inplace=True,
            )

        # rename index values for index 'interaction'
        if interact_abbr:
            df_s2.reset_index(level="interaction", drop=False, inplace=True)
            for key, value in interact_abbr.items():
                df_s2.replace({"interaction": key}, value, inplace=True)
            df_s2.set_index(keys="interaction", drop=True, append=True, inplace=True)

        return df_s2
