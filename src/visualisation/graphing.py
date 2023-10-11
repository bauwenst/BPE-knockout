"""
Copyright (C) 2021-2023 Thomas Bauwens
This code was written across multiple research papers at university.
Any copy of this code must contain this copyright notice.
It may only be distributed with explicit permission from the author.

TODO:
    - Histogram should be a special case of MultiHistogram. The graphing code is basically duplicated
      and very annoying to maintain in sync.
"""
from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple

import itertools
from pathlib import Path
import numpy as np
import json
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.font_manager
from matplotlib import rc

rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)

from matplotlib import colors as mcolors

MPL_COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
NICE_COLORS = [
    MPL_COLORS.get("r"), MPL_COLORS.get("g"), MPL_COLORS.get("b"),
    MPL_COLORS.get("lime"), MPL_COLORS.get("darkviolet"), MPL_COLORS.get("gold"),
    MPL_COLORS.get("cyan"), MPL_COLORS.get("magenta")
]

ASPECT_RATIO_SIZEUP = 2
DEFAULT_ASPECT_RATIO = (4,3)
DEFAULT_GRIDWIDTH = 0.5

from src.auxiliary.paths import PATH_DATA_OUT
PATH_FIGURES = PATH_DATA_OUT / "figures"
PATH_FIGURES.mkdir(exist_ok=True)
PATH_RAW = PATH_FIGURES / "raw"
PATH_RAW.mkdir(exist_ok=True)


def getColours():
    return list(NICE_COLORS)


def newFigAx(aspect_ratio: Tuple[float,float]):
    return plt.subplots(figsize=(ASPECT_RATIO_SIZEUP*aspect_ratio[0], ASPECT_RATIO_SIZEUP*aspect_ratio[1]))


class Diagram(ABC):

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def safeFigureWrite(stem: str, suffix: str, figure, show=False):
        """
        Write a matplotlib figure to a file. For best results, use suffix=".pdf".
        The write is "safe" because it searches for a file name that doesn't exist yet, instead of overwriting.
        """
        if show:
            plt.show()  # Somtimes matplotlib hangs on savefig, and showing the figure can "slap the TV" to get it to work.
        print(f"Writing figure {stem} ...")
        modifier = 0
        while (PATH_FIGURES / (stem + "_" + str(modifier) + suffix)).exists():
            modifier += 1
        figure.savefig(PATH_FIGURES.as_posix() + "/" + stem + "_" + str(modifier) + suffix, bbox_inches='tight')

    @staticmethod
    def safeDatapointWrite(stem: str, data: dict):
        """
        Write a json of data points to a file. Also safe.
        """
        print(f"Writing json {stem} ...")
        modifier = 0
        while (PATH_RAW / (stem + "_" + str(modifier) + ".json")).exists():
            modifier += 1
        with open(PATH_RAW / (stem + "_" + str(modifier) + ".json"), "w") as file:
            json.dump(data, file)

    @abstractmethod
    def clear(self):
        """
        Reset all data in the object.
        """
        pass

    # @abstractmethod  Should be abstract, but I don't have the time to implement it for all classes currently
    def load(self, json_path: Path):
        """
        Load object from a json written by safeDatapointWrite.
        """
        pass


class Graph(Diagram):
    """
    2D line plot.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.series = dict()

    def initSeries(self, series_name: str):
        self.series[series_name] = ([], [])

    def add(self, series_name: str, x, y):
        """
        Add a single datapoint to the series (line) with the given label.
        """
        if type(x) == str or type(y) == str:
            print("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return

        if series_name not in self.series:
            self.initSeries(series_name)
        self.series[series_name][0].append(x)
        self.series[series_name][1].append(y)

    def addMany(self, series_name: str, xs: Sequence, ys: Sequence):
        if series_name not in self.series:
            self.initSeries(series_name)
        self.series[series_name][0].extend(xs)
        self.series[series_name][1].extend(ys)

    def commit(self, aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label="", legend_position="lower right",
               do_points=True, initial_style_idx=0,
               grid_linewidth=DEFAULT_GRIDWIDTH, curve_linewidth=1, optional_line_at_y=None,
               y_lims=None, x_tickspacing=None, y_tickspacing=None, logx=False, logy=False,
               slap_the_tv=False,
               restorable=True, only_for_return=False, existing_figax: tuple=None):
        """
        Render a figure based on the added data.
        Also stores the data to a JSON file (see save()).

        Since figure rendering can error due to the LaTeX compiler (for example, because your axis labels use unicode
        instead of LaTeX commands), the entire implementation is wrapped in a try-except.
        Yes, I had to find out the hard way by losing a 6-hour render.
        """
        if restorable:
            self.save()

        try:
            # The graph style is a tuple (col, line, marker) that cycles from front to back:
            #   - red solid dot, blue solid dot, green solid dot
            #   - red dashed dot, blue dashed dot, green dashed dot
            #   - ...
            #   - red solid x, blue solid x, green solid x
            #   - ...
            colours = getColours()
            line_styles = ["-", "--", ":"]
            line_markers = [".", "x", "+"] if do_points else [""]
            line_styles = list(itertools.product(line_markers, line_styles, colours))

            if existing_figax is None:
                fig, main_ax = newFigAx(aspect_ratio)
            else:
                fig, main_ax = existing_figax
            main_ax.grid(True, which='both', linewidth=grid_linewidth)
            main_ax.axhline(y=0, color='k', lw=0.5)

            style_idx = initial_style_idx
            for name, samples in self.series.items():
                marker, line, colour = line_styles[style_idx % len(line_styles)]
                style = marker + line

                if logx and logy:
                    main_ax.loglog(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                elif logx:
                    main_ax.semilogx(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                elif logy:
                    main_ax.semilogy(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)
                else:
                    main_ax.plot(samples[0], samples[1], style, c=colour, label=name, linewidth=curve_linewidth)

                style_idx += 1

            if optional_line_at_y is not None:
                main_ax.hlines(optional_line_at_y,
                               min([min(tup[0]) for tup in self.series.values()]),
                               max([max(tup[0]) for tup in self.series.values()]), colors='b', linestyles='dotted')

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)
            main_ax.legend(loc=legend_position)

            if y_lims:
                main_ax.set_ylim(y_lims[0], y_lims[1])

            if x_tickspacing:
                x_min, x_max = main_ax.get_xlim()
                main_ax.set_xticks(np.arange(0, x_max, x_tickspacing))

            if y_tickspacing:
                y_min, y_max = main_ax.get_ylim()
                main_ax.set_yticks(np.arange(0, y_max, y_tickspacing))

            if y_lims:  # Yes, twice. Don't ask.
                main_ax.set_ylim(y_lims[0], y_lims[1])

            if not only_for_return:
                Diagram.safeFigureWrite(self.name, ".pdf", fig, show=slap_the_tv)
            return fig, main_ax
        except Exception as e:
            print(f"Error writing figure '{self.name}'; probably a LaTeX thing. "
                  f"Your datapoints were cached, but you may have to go into the JSON to delete illegal characters in the series names.")
            print("Here's the text:", "\"", e, "\"")

    def merge_commit(self, other_graph: "Graph", aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label_left="", y_label_right=""):
        """
        Draw two graphs onto the same figure. The x axis is shared, the first graph's y axis is on the left, and the
        second graph's y axis is on the right.
        """
        merged_name = self.name + "+" + other_graph.name
        Diagram.safeDatapointWrite(merged_name, {self.name: self.series, other_graph.name: other_graph.series})

        ######## ALREADY IN .COMMIT
        # First graph
        colours = getColours()
        fig, ax1 = newFigAx(aspect_ratio)
        ax1.grid(True, which='both')
        ax1.axhline(y=0, color='k', lw=0.5)

        for name, samples in self.series.items():
            ax1.plot(samples[0], samples[1], c=colours.pop(0), marker=".", linestyle="-", label=name)
        ########

        # Second graph
        ax2 = ax1.twinx()  # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.

        ######## ALREADY IN .COMMIT
        for name, samples in other_graph.series.items():
            ax2.plot(samples[0], samples[1], c=colours.pop(0), marker=".", linestyle="-", label=name)

        # Labels
        if x_label:
            ax1.set_xlabel(x_label)
        if y_label_left:
            ax1.set_ylabel(y_label_left)
        if y_label_right:
            ax2.set_ylabel(y_label_right)
        #########

        # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
        legend_1 = ax1.legend(loc='upper right')
        legend_1.remove()
        ax2.legend(loc='lower right')
        ax2.add_artist(legend_1)

        Diagram.safeFigureWrite(merged_name, ".pdf", fig)

    def merge_commit2(self, fig, ax1, other_graph: "Graph", **second_commit_kwargs):
        """
        The signature is vague because this is the only way to abstract the process of having a twin x-axis.
        Basically, here are the options:
            1. The user commits graph 1 without saving, takes a twinx for the ax, passes that into the commit of graph 2,
               and then passes the fig,ax1,ax2 triplet into a third function that adds the legends and saves.
            2. Same first step, but do the rest in this function, sadly not having autocompletion for the second graph's
               commit arguments.
            3. Same, except copy all the options from .commit() and pick the ones to re-implement here (e.g. styling, y
               limits ...).
            4. Do everything inside this function by copying the signature from .commit() twice (once per graph).

        The last two are really bad design, and really tedious to maintain.
        """
        name = "merged_(" + self.name + ")_(" + other_graph.name + ")"

        # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.
        ax2 = ax1.twinx()

        # Modify ax2 in-place.
        other_graph.commit(**second_commit_kwargs, existing_figax=(fig,ax2), initial_style_idx=len(self.series), only_for_return=True, restorable=True)

        # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
        legend_1 = ax1.legend(loc='upper right')
        legend_1.remove()
        ax2.legend(loc='lower right')
        ax2.add_artist(legend_1)

        # At last, save.
        Diagram.safeFigureWrite(name, ".pdf", fig)

    def clear(self):
        self.series = dict()

    def save(self):
        """
        Save to a file. This happens automatically on a commit(), just for good measure.
        """
        Diagram.safeDatapointWrite(self.name, self.series)

    def load(self, json_path: Path):
        """
        Restore a file by extending the series off this object.
        """
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r") as handle:
            series: dict = json.load(handle)

        for name, tup in series.items():
            if len(tup) != 2 or len(tup[0]) != len(tup[1]):
                raise ValueError(
                    "Graph data corrupted: either there aren't two tracks for a series, or they don't match in length.")

            self.addMany(name, tup[0], tup[1])

    @staticmethod
    def qndLoadAndCommit(json_path: Path):
        """
        Quick-and-dirty method to load in the JSON data of a graph and commit it without
        axis formatting etc. Useful when rendering with commit() failed but your JSON was
        saved and you want to get a rough idea of what came out of it.

        Strips the part of the stem at the end that starts with "_".
        """
        raw_name = json_path.stem
        g = Graph(raw_name[:raw_name.rfind("_")])
        g.load(json_path)
        g.commit(restorable=False)


class Bars(Diagram):
    """
    Multi-bar chart. Produces a chart with groups of bars on them: each group has the same amount of bars and the
    colours of the bars are in the same order per group.

    All the bars of the same colour are considered the same family. All the families must have the same amount of
    bars, which is equal to the amount of groups.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.series = dict()

    def add(self, bar_slice_family: str, height: float):
        if bar_slice_family not in self.series:
            self.series[bar_slice_family] = []
        self.series[bar_slice_family].append(height)

    def addMany(self, bar_slice_family: str, heights: Sequence[float]):
        if bar_slice_family not in self.series:
            self.series[bar_slice_family] = []
        self.series[bar_slice_family].extend(heights)

    def commit(self, group_names: Sequence[str], bar_width: float, group_spacing: float, y_label: str="",
               restorable=True, diagonal_labels=True, aspect_ratio=DEFAULT_ASPECT_RATIO,
               y_tickspacing: float=None, log_y: bool=False):
        if restorable:
            Diagram.safeDatapointWrite(self.name, self.series)

        fig, main_ax = newFigAx(aspect_ratio)
        main_ax: plt.Axes

        colours = getColours()
        group_locations = None
        for i, (bar_slice_family, slice_heights) in enumerate(self.series.items()):
            group_locations = group_spacing * np.arange(len(slice_heights))
            main_ax.bar(group_locations + bar_width*i, slice_heights, color=colours.pop(0), width=bar_width,
                        label=bar_slice_family)

        # X-axis
        main_ax.set_xticks(group_locations + bar_width * len(self.series) / 2 - bar_width / 2)
        main_ax.set_xticklabels(group_names, rotation=45*diagonal_labels, ha="right" if diagonal_labels else "center")
        # main_ax.set_yticks(np.arange(0, 1.1, 0.1))

        # Y-axis
        main_ax.set_ylabel(y_label)
        # main_ax.set_ylim(0, 1)
        if log_y:
            main_ax.set_yscale("log")

        # Grid
        main_ax.set_axisbelow(True)  # Put grid behind the bars.
        main_ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
        main_ax.legend()

        Diagram.safeFigureWrite(self.name, ".pdf", fig)

    def clear(self):
        self.series = dict()


class Histogram(Diagram):
    """
    Cross between a graph and a bar chart.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.x_values = []

    def add(self, x_value: float):
        self.x_values.append(x_value)

    def addMany(self, values: Sequence[float]):
        self.x_values.extend(values)

    def load(self, json_path: Path):
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r") as handle:
            data: dict = json.load(handle)

        data = data.get("x_values")
        if data is None:
            raise ValueError("Histogram data corrupted.")

        self.addMany(data)

    def clear(self):
        self.x_values = []

    def toDataframe(self):
        rows = []
        for v in self.x_values:
            rows.append({"value": v})
        df = pd.DataFrame(rows)
        return df

    def commit_matplotlib(self, width: float, x_label="", y_label="", restorable=True, aspect_ratio=DEFAULT_ASPECT_RATIO):
        if restorable:
            Diagram.safeDatapointWrite(self.name, {"x_values": self.x_values})

        try:
            fig, main_ax = newFigAx(aspect_ratio)
            main_ax.hist(self.x_values, bins=int((max(self.x_values) - min(self.x_values)) // width))  # You could do this, but I don't think the ticks will then be easy to set.

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)

            Diagram.safeFigureWrite(self.name, ".pdf", fig)
        except:
            print("Histogram failed, but data saved.")

    def commit_histplot(self, binwidth: float=1, x_tickspacing: float=1, center_ticks=False,
                        x_label: str = "", y_label: str = "", aspect_ratio=DEFAULT_ASPECT_RATIO,
                        restorable=True, x_lims: Tuple[int,int]=None,
                        relative_counts: bool=False, average_over_bin: bool=False,
                        do_kde=True, kde_smoothing=True,
                        border_colour=None, fill_colour=None,  # Note: colour=None means "use default colour", not "use no colour".
                        log_x=False, log_y=False, y_tickspacing: float=None,
                        **seaborn_args):
        """
        :param x_lims: left and right bounds. Either can be None to make them automatic. These bound are the edge of
                       the figure; if you have a bar from x=10 to x=11 and you set the right bound to x=10, then you
                       won't see the bar but you will see the x=10 tick.
        :param center_ticks: Whether to center the ticks on the bars. The bars at the minimal and maximal x_lim are
                             only half-visible.
        """
        if restorable:
            Diagram.safeDatapointWrite(self.name, {"x_values": self.x_values})

        if relative_counts:
            if average_over_bin:
                mode = "density"  # Total area is 1.
            else:
                mode = "percent"  # Total area is 100.
        else:
            if average_over_bin:
                mode = "frequency"
            else:
                mode = "count"

        df = self.toDataframe()

        fig, ax = newFigAx(aspect_ratio)
        if not log_x:
            sns.histplot(df, x="value", ax=ax, kde=do_kde, discrete=center_ticks, kde_kws={"bw_adjust": 10} if kde_smoothing else None,
                         binwidth=binwidth, binrange=(math.floor(df["value"].min()/binwidth) * binwidth, math.ceil(df["value"].max() / binwidth) * binwidth),
                         stat=mode, color=fill_colour, edgecolor=border_colour,
                         **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077
        else:
            sns.histplot(df, x="value", ax=ax, log_scale=True,
                         stat=mode, color=fill_colour, edgecolor=border_colour,
                         **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label + r" [\%]" * (mode == "percent" and y_label != ""))
        if x_lims:
            if x_lims[0] is not None and x_lims[1] is not None:
                ax.set_xlim(x_lims[0],x_lims[1])
            elif x_lims[0] is not None:
                ax.set_xlim(left=x_lims[0])
            elif x_lims[1] is not None:
                ax.set_xlim(right=x_lims[1])
            else:  # You passed (None,None)...
                pass

        if not log_x:
            ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
            ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

        if not log_y:
            if y_tickspacing:
                ax.yaxis.set_major_locator(tkr.MultipleLocator(y_tickspacing))
                ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
        else:
            ax.set_yscale("log")

        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
        Diagram.safeFigureWrite(self.name + "_histplot", ".pdf", fig)

    def commit_boxplot(self, x_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.boxplot(df, x="value", showmeans=True, orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([""])
        Diagram.safeFigureWrite(self.name + "_boxplot", ".pdf", fig)

    def commit_violin(self, x_label="", y_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.violinplot(df, x="value", orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([y_label], rotation=90, va='center')
        Diagram.safeFigureWrite(self.name + "_violinplot", ".pdf", fig)


class MultiHistogram(Diagram):

    def __init__(self, name: str):
        super().__init__(name)
        self.x_values = dict()

    def add(self, series_name: str, x_value: float):
        if series_name not in self.x_values:
            self.x_values[series_name] = []
        self.x_values[series_name].append(x_value)

    def addMany(self, series_name: str, values: Sequence[float]):
        if series_name not in self.x_values:
            self.x_values[series_name] = []
        self.x_values[series_name].extend(values)

    def commit(self, width: float, x_label="", y_label="", restorable=True, aspect_ratio=DEFAULT_ASPECT_RATIO):
        if restorable:
            Diagram.safeDatapointWrite(self.name, self.x_values)

        try:
            fig, main_ax = newFigAx(aspect_ratio)
            for name,x_values in self.x_values.items():
                main_ax.hist(x_values, bins=int((max(x_values) - min(x_values)) // width))  # You could do this, but I don't think the ticks will then be easy to set.

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)

            Diagram.safeFigureWrite(self.name, ".pdf", fig)
        except:
            print("Histogram failed" + ", but data saved"*restorable + ".")

    def commit_histplot(self, binwidth: float=1, x_tickspacing: float=1,
                        x_label: str="", y_label: str="", aspect_ratio=DEFAULT_ASPECT_RATIO, restorable=True,
                        x_lims: Tuple[int,int]=None, relative_counts: bool=False,
                        border_colour=None, center_ticks=False,
                        do_kde=True, y_tickspacing: float=None,
                        **seaborn_args):
        if restorable:
            Diagram.safeDatapointWrite(self.name, self.x_values)

        CLASS_NAME = "klasse"

        try:
            # You would think all it takes is pd.Dataframe(dictionary), but that's the wrong way to do it.
            # If you do it that way, you are pairing up the i'th value of each family as if it belongs to one
            # object. This is not correct, and crashes if you have different sample amounts per family.
            rows = []  # For speed, instead of appending to a dataframe, make a list of rows as dicts. https://stackoverflow.com/a/17496530/9352077
            for name, x_values in self.x_values.items():
                for v in x_values:
                    rows.append({"value": v, CLASS_NAME: name})
            df = pd.DataFrame(rows)
            print(df.groupby(CLASS_NAME).describe())

            fig, ax = newFigAx(aspect_ratio)
            ax: plt.Axes
            sns.histplot(df, x="value", hue=CLASS_NAME,  # x and hue args: https://seaborn.pydata.org/tutorial/distributions.html
                         ax=ax, kde=do_kde, kde_kws={"bw_adjust": 10},  # Btw, do not use displot: https://stackoverflow.com/a/63895570/9352077
                         binwidth=binwidth, binrange=(math.floor(df["value"].min()/binwidth)*binwidth,math.ceil(df["value"].max()/binwidth)*binwidth),
                         edgecolor=border_colour, discrete=center_ticks,
                         stat="percent" if relative_counts else "count", common_norm=False,
                         **seaborn_args)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label + r" [\%]"*(relative_counts and y_label != ""))
            if x_lims:
                ax.set_xlim(x_lims[0], x_lims[1])

            # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
            ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
            ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
            if y_tickspacing:
                ax.yaxis.set_major_locator(tkr.MultipleLocator(y_tickspacing))
                ax.yaxis.set_major_formatter(tkr.ScalarFormatter())

            # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
            Diagram.safeFigureWrite(self.name + "_histplot", ".pdf", fig)

        except:
            print("MultiHistogram failed to save.")

    def commit_boxplot(self, value_axis_label: str= "", class_axis_label: str= "",
                       aspect_ratio=DEFAULT_ASPECT_RATIO, restorable=True,
                       log=False, horizontal=False, iqr_limit=1.5,
                       value_tickspacing=None):
        """
        Draws multiple boxplots side-by-side.
        Note: the "log" option doesn't just stretch the value axis, because in
        that case you will get a boxplot on skewed data and then stretch that bad
        boxplot. Instead, this method applies log10 to the values, then computes
        the boxplot, and plots it on a regular axis.
        """
        if restorable:
            Diagram.safeDatapointWrite(self.name, self.x_values)

        try:
            rows = []
            for name, x_values in self.x_values.items():
                for v in x_values:
                    if log:
                        rows.append({"value": np.log10(v), "class": name})
                    else:
                        rows.append({"value": v, "class": name})
            df = pd.DataFrame(rows)
            print(df.groupby("class").describe())

            fig, ax = newFigAx(aspect_ratio)
            ax: plt.Axes

            # Format outliers (https://stackoverflow.com/a/35133139/9352077)
            flierprops = {
                # "markerfacecolor": '0.75',
                # "linestyle": 'none',
                "markersize": 0.1,
                "marker": "."
            }

            if log and value_axis_label:
                value_axis_label = "$\log_{10}($" + value_axis_label + "$)$"

            if horizontal:
                sns.boxplot(df, x="value", y="class",
                            ax=ax, linewidth=0.5, flierprops=flierprops)
                ax.set_xlabel(value_axis_label)
                ax.set_ylabel(class_axis_label)
            else:
                sns.boxplot(df, x="class", y="value",
                            ax=ax, linewidth=0.5, flierprops=flierprops,
                            whis=iqr_limit)
                ax.set_xlabel(class_axis_label)
                ax.set_ylabel(value_axis_label)

                if value_tickspacing:
                    # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
                    import matplotlib.ticker as tck
                    ax.yaxis.set_major_locator(tck.MultipleLocator(value_tickspacing))
                    ax.yaxis.set_major_formatter(tck.ScalarFormatter())

            # if x_lims:
            #     ax.set_xlim(x_lims[0], x_lims[1])
            #
            # # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            # ax.set_axisbelow(True)
            # ax.grid(True, axis="y")
            Diagram.safeFigureWrite(self.name + "_boxplot", ".pdf", fig)

        except Exception as e:
            print("MultiHistogram failed to save.", e)

    def load(self, json_path: Path):
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r") as handle:
            data: dict = json.load(handle)

        for name, values in data.items():
            if not isinstance(values, list):
                raise ValueError("Histogram data corrupted.")
            self.addMany(name, values)

    def clear(self):
        self.x_values = dict()


class ScatterPlot(Diagram):

    def __init__(self, name: str):
        super().__init__(name)
        self.families = dict()

    def addPointsToFamily(self, family_name: str, xs, ys):
        """
        Unlike the other diagram types, it seems justified to add scatterplot points in bulk.
        Neither axis is likely to represent time, so you'll probably have many points available at once.
        """
        if family_name not in self.families:
            self.families[family_name] = ([], [])
        self.families[family_name][0].extend(xs)
        self.families[family_name][1].extend(ys)

    def commit(self, x_label="", y_label="",
               x_lims=None, y_lims=None, logx=False, logy=False, x_tickspacing=None,
               family_colours=None, family_sizes=None, randomise_markers=False,
               save_to_file=True, aspect_ratio: Tuple[float,float]=DEFAULT_ASPECT_RATIO, grid=False, legend=False):
        fig, ax = newFigAx(aspect_ratio)
        ax: plt.Axes

        if logx:
            ax.set_xscale("log")  # Needed for a log scatterplot. https://stackoverflow.com/a/52573929/9352077
            ax.xaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))  # See comment under https://stackoverflow.com/q/76285293/9352077
            ax.xaxis.set_major_formatter(tkr.LogFormatterSciNotation())
        else:
            if x_tickspacing:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(tkr.LogLocator(base=10, numticks=999))
            ax.yaxis.set_major_formatter(tkr.LogFormatterSciNotation())

        if logx and logy:  # Otherwise you have a skewed view of horizontal vs. vertical distances.
            ax.set_aspect("equal")

        if family_colours is None:
            family_colours = dict()
        if family_sizes is None:
            family_sizes = dict()

        markers = {".", "^", "+", "s"}
        # cols = getColours()
        cols = plt.cm.rainbow(np.linspace(0, 1, len(self.families)))  # Equally spaced rainbow colours.
        scatters = []
        names    = []
        for idx, tup in enumerate(sorted(self.families.items(), reverse=True)):
            name, family = tup
            m = markers.pop() if randomise_markers else "."
            c = family_colours.get(name, cols[idx])
            s = family_sizes.get(name, 40)
            result = ax.scatter(family[0], family[1], marker=m, linewidths=0.05, color=c, s=s)
            scatters.append(result)
            names.append(name)

        if x_lims:
            ax.set_xlim(x_lims[0], x_lims[1])
        if y_lims:
            ax.set_ylim(y_lims[0], y_lims[1])

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        if grid:
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=DEFAULT_GRIDWIDTH)

        if legend:
            ax.legend(scatters, names, loc='upper left', markerscale=10, ncol=2)  # https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend

        # Diagram.safeDatapointWrite(self.name, self.families)
        if save_to_file:
            Diagram.safeFigureWrite(self.name, ".pdf", fig)
        return fig, ax

    def clear(self):
        self.families = dict()

    def copy(self, new_name: str):
        new_plot = ScatterPlot(new_name)
        for name, values in self.families.items():
            new_plot.addPointsToFamily(name, values[0].copy(), values[1].copy())
        return new_plot


def arrow(ax: plt.Axes, start_point,
          end_point):  # FIXME: I want TikZ's stealth arrows, but this only seems possible in Matplotlib's legacy .arrow() interface (which doesn't keep its head shape properly): https://stackoverflow.com/a/43379608/9352077
    """
    Matplotlib's arrow interface is impossibly complicated. This simplifies that.
    Based on:
    https://stackoverflow.com/a/52613154/9352077
    """
    prop = {
        "arrowstyle": "->",
        # "overhang": 0.2,
        # "headwidth": 0.4,
        # "headlength": 0.8,
        "shrinkA": 0,
        "shrinkB": 0,
        "linewidth": 0.4
    }
    ax.annotate("", xy=end_point, xytext=start_point, arrowprops=prop)


def simpleTable(TP, FN, FP, TN):
    tp_len = len(str(TP))
    tn_len = len(str(TN))
    fp_len = len(str(FP))
    fn_len = len(str(FN))

    column_1_size = max(tp_len, fp_len)
    column_2_size = max(fn_len, tn_len)

    s = "TP: " + " " * (column_1_size - tp_len) + f"{TP} | FN: " + " " * (column_2_size - fn_len) + f"{FN}\n" + \
        "FP: " + " " * (column_1_size - fp_len) + f"{FP} | TN: " + " " * (column_2_size - tn_len) + f"{TN}"
    return s


def latexTable(name, TP, FN, FP, TN, F1frac, MSEfrac, MSE_label: str = "MSE"):
    s = r"\begin{tabular}{rrcc}" + "\n" + \
        r"                     &                          & \multicolumn{2}{c}{$f(\vec X)$} \\" + "\n" + \
        r"      \multicolumn{2}{r}{" + name + r"}                   & \multicolumn{1}{|c|}{$P$}  & $N$ \\ \cline{2-4} " + "\n" + \
        r"      \multirow{2}{*}{$Y$} & \multicolumn{1}{r|}{$P$} & " + str(TP) + " & " + str(
        FN) + r"   \\ \cline{2-2}" + "\n" + \
        r"                           & \multicolumn{1}{r|}{$N$} & " + str(FP) + " & " + str(
        TN) + r"\\ \cline{2-4}" + "\n" + \
        r"                           & \multicolumn{3}{l}{$F_1 = " + str(
        round(F1frac * 100, 2)) + r"\%$}                        \\" + "\n" + \
        r"                           & \multicolumn{3}{l}{${\scalebox{0.75}{$" + MSE_label + r"$}} = " + str(
        round(MSEfrac * 100, 2)) + r"\%$}" + "\n" + \
        r"\end{tabular}"

    return s


# --- --- --- --- --- ---
def example():
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    fig, main_ax = plt.subplots()
    main_ax.plot(t, t, 'r--')
    main_ax.plot(t, t ** 2, 'bs')
    main_ax.plot(t, t ** 3, 'g^')
    # main_ax.set_xlim(0, 1)
    # main_ax.set_ylim(1.1 * np.min(s), 2 * np.max(s))
    main_ax.set_xlabel('time (s)')
    main_ax.set_ylabel('current (nA)')
    main_ax.set_title('Gaussian colored noise')

    fig.savefig("./figures/test.pdf", bbox_inches='tight')


def example2():
    data = [[30, 25, 50, 20],
            [40, 23, 51, 17],
            [35, 22, 45, 19]]
    X = np.arange(4)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X + 0.00, data[0], color='b', width=0.25)
    ax.bar(X + 0.25, data[1], color='g', width=0.25)
    ax.bar(X + 0.50, data[2], color='r', width=0.25)

    ax.set_xticks(X + 0.25)
    ax.set_xticklabels(('aaaaaaaX1', 'X2', 'bbbbX3', 'X4'), rotation=45, ha="right")

    fig.savefig("./figures/bars.pdf", bbox_inches='tight')


def example_multihistogram():
    import numpy.random as npr
    h = MultiHistogram("example")
    h.addMany("a", npr.normal(loc=0.0, scale=1.0, size=50))
    h.addMany("b", npr.normal(loc=4.0, scale=3.0, size=30))

    h.commit_histplot(restorable=False, x_tickspacing=1, binwidth=0.5)