"""
Copyright (C) 2021-2023 Thomas Bauwens
This code was written across multiple research papers at university.
Any copy of this code must contain this copyright notice.
It may only be distributed with explicit permission from the author.

TODO:
    - Histogram is a subclass of MultiHistogram, but has some methods that are "more sophisticated" that should be
      integrated into MultiHistogram.
    - The JSON and PDF names are checked for availability independently, so JSON _0 can correspond to PDF _2 etc.
"""
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Union, Sequence, Tuple, List, Dict

import itertools
from pathlib import Path
import numpy as np
import json
import pandas as pd
import math
import time

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.font_manager
from matplotlib import rc

# Enable LaTeX support
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)

# Colour setup
from matplotlib import colors as mcolors
MPL_COLORS = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
NICE_COLORS = [
    MPL_COLORS.get("r"), MPL_COLORS.get("g"), MPL_COLORS.get("b"),
    MPL_COLORS.get("lime"), MPL_COLORS.get("darkviolet"), MPL_COLORS.get("gold"),
    MPL_COLORS.get("cyan"), MPL_COLORS.get("magenta")
]
def getColours():
    return list(NICE_COLORS)

# Graphical defaults
ASPECT_RATIO_SIZEUP = 2
DEFAULT_ASPECT_RATIO = (4,3)
DEFAULT_GRIDWIDTH = 0.5

# Strings that appear in graphs (change to language of choice)
LEGEND_TITLE_CLASS = "class"

# Path setup (if you ported this file from somewhere, you probably have to change the below import)
from src.auxiliary.paths import PATH_DATA_OUT
PATH_FIGURES = PATH_DATA_OUT / "figures"
PATH_FIGURES.mkdir(exist_ok=True)
PATH_RAW = PATH_FIGURES / "raw"
PATH_RAW.mkdir(exist_ok=True)


class PathHandling:
    """
    Appends _0, _1, _2 ... to file stems so that there are no collisions in the file system.
    """

    @staticmethod
    def makePath(folder: Path, stem: str, modifier: int, suffix: str) -> Path:
        return folder / f"{stem}_{modifier}{suffix}"

    @staticmethod
    def getSafeModifier(folder: Path, stem: str, suffix: str) -> int:
        modifier = 0
        path = PathHandling.makePath(folder, stem, modifier, suffix)
        while path.exists():
            modifier += 1
            path = PathHandling.makePath(folder, stem, modifier, suffix)
        return modifier

    @staticmethod
    def getSafePath(folder: Path, stem: str, suffix: str) -> Path:
        return PathHandling.makePath(folder, stem, PathHandling.getSafeModifier(folder, stem, suffix), suffix)

    @staticmethod
    def getHighestAlias(folder: Path, stem: str, suffix: str) -> Union[Path, None]:
        safe_modifier = PathHandling.getSafeModifier(folder, stem, suffix)
        if safe_modifier == 0:
            return None
        return PathHandling.makePath(folder, stem, safe_modifier-1, suffix)


def newFigAx(aspect_ratio: Tuple[float,float]) -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(ASPECT_RATIO_SIZEUP*aspect_ratio[0], ASPECT_RATIO_SIZEUP*aspect_ratio[1]))


class CacheMode(Enum):
    """
    Diagram data is stored in JSON files if desired by the user, and read back if desired by the user.
    The user likely wants one of the behaviours grouped below.

    | Exists | Read | Write | What does this mean?                                                  |
    | ---    | ---  | ---   | ---                                                                   |
    | no     | no   | no    | always compute never cache; good for examples or easy experiments     |
    | yes    | no   | no    | idem                                                                  |

    | no     | no   | yes   | equivalent to first run; always-write                                 |
    | yes    | no   | yes   | re-compute and refresh cache; always-write                            |

    | no     | yes  | yes   | most common situation during the first run                            |
    | yes    | yes  | no    | most common situation after 1 run                                     |

    | no     | yes  | no    | cache miss that you won't correct (can't know that it'll miss though) |

    | yes    | yes  | yes   | makes no sense because you are writing what you just read             |
    """
    # Don't read, don't write. Pretend like there is no cache.
    NONE = 1
    # Don't read, but always write, regardless of whether a file already exists. Useful for prototyping. Note: there is no "do read and always refresh" because if you want to refresh using what you read, you should use a different file.
    WRITE_ONLY = 2
    # Read but never write. Useful in a very select amount of cases, e.g. testing the reading system.
    READ_ONLY = 3
    # Read, and only write *IF* that fails. Note: there is no "don't read, but still check whether file exists and only write *if* missing" mode.
    IF_MISSING = 4


class Diagram(ABC):

    def __init__(self, name: str, caching: CacheMode=CacheMode.NONE):
        """
        Constructs a Diagram object with a name (for file I/O) and space to store data.
        The reason why the subclasses don't have initialisers is two-fold:
            1. They all looked like
                def __init__(self, name: str, use_cached: bool):
                    self.some_kind_of_dictionary = dict()
                    super().__init__(name, use_cached)
            2. It's not proper OOP to put the superclass's initialiser after the subclass's initialiser, but the way
               this initialiser is written, it is inevitable: it calls self.load() which accesses the subclass's fields,
               and if those fields are defined by the subclass, you get an error.
               While it is allowed in Python (https://stackoverflow.com/q/45722427/9352077), I avoid it.

        :param name: The file stem to be used for everything produced by this object.
        :param caching: Determines how the constructor will attempt to find the most recent data file matching the name
                        and load those data into the object. Also determines whether commit methods will store data.
        """
        self.name = name
        self.data = dict()  # All figure classes are expected to store their data in a dictionary.

        self.needs_computation = (caching == CacheMode.NONE or caching == CacheMode.WRITE_ONLY)
        self.will_be_stored    = (caching == CacheMode.WRITE_ONLY)
        if caching == CacheMode.READ_ONLY or caching == CacheMode.IF_MISSING:
            already_exists = False

            # Find file, and if you find it, try to load from it.
            cache_path = PathHandling.getHighestAlias(PATH_RAW, self.name, ".json")
            if cache_path is not None:  # Possible cache hit
                try:
                    self.load(cache_path)
                    print(f"Successfully preloaded data for diagram '{self.name}'.")
                    already_exists = True
                except Exception as e:
                    print("Could not load cached diagram:", e)

            if not already_exists:  # Cache miss
                self.needs_computation = True
                self.will_be_stored    = (caching == CacheMode.IF_MISSING)

    ### STATIC METHODS (should only be used if the non-static methods don't suffice)

    @staticmethod
    def safeFigureWrite(stem: str, suffix: str, figure, show=False):
        """
        Write a matplotlib figure to a file. For best results, use suffix=".pdf".
        The write is "safe" because it searches for a file name that doesn't exist yet, instead of overwriting.
        """
        if show:
            plt.show()  # Somtimes matplotlib hangs on savefig, and showing the figure can "slap the TV" to get it to work.
        print(f"Writing figure {stem} ...")
        figure.savefig(PathHandling.getSafePath(PATH_FIGURES, stem, suffix).as_posix(), bbox_inches='tight')

    @staticmethod
    def safeDatapointWrite(stem: str, data: dict):
        """
        Write a json of data points to a file. Also safe.
        """
        print(f"Writing json {stem} ...")
        with open(PathHandling.getSafePath(PATH_RAW, stem, ".json"), "w") as file:
            json.dump(data, file)

    ### IMPLEMENTATIONS

    def exportToPdf(self, fig, stem_suffix: str=""):
        Diagram.safeFigureWrite(stem=self.name + stem_suffix, suffix=".pdf", figure=fig)

    def save(self, metadata: dict=None):
        Diagram.safeDatapointWrite(stem=self.name, data={
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or dict(),
            "data": self._save()
        })

    def load(self, json_path: Path):
        if not json_path.suffix == ".json" or not json_path.is_file():
            raise ValueError(f"Cannot open JSON: file {json_path.as_posix()} does not exist.")

        with open(json_path, "r") as handle:
            object_as_dict: dict = json.load(handle)

        if "data" not in object_as_dict:
            raise KeyError(f"Cannot read JSON file: 'data' key missing.")

        self._load(object_as_dict["data"])

    ### INSTANCE METHODS (can be overridden for complex objects whose data dictionaries aren't JSON-serialisable and/or have more state and fields)

    def clear(self):
        """
        Reset all data in the object.
        """
        self.data = dict()

    def _save(self) -> dict:
        """
        Serialise the object.
        """
        return self.data

    def _load(self, saved_data: dict):
        """
        Load object from the dictionary produced by _save().
        It is recommended to override this with appropriate sanity checks!
        """
        self.data = saved_data


class ProtectedData:
    """
    Use this in any commit method to protect data like this:
        def commit(...):
            with ProtectedData(self):
                ...
                self.exportToPdf(fig)
    If the user demanded that those data be cached, this will be done immediately.
    If an error occurs before (or during) the export, caching is also done, AND the exception is swallowed;
    this way, if you are calling multiple commits in a row, the other commits can still be reached even if one fails.

    Why not just wrap this commit method in a try...except in the abstract class? Two reasons:
        1. A commit method has many graphical arguments, and they differ from subclass to subclass. The user would have
           to call the wrapper, but there is no way to let PyCharm autocomplete the arguments in that case. A wrapper
           would need *args and **kwargs, and if you use a decorator, PyCharm automatically shows *args and **kwargs.
        2. Some classes have different commit methods (e.g. histograms to violin plot vs. boxplot). Hence, the abstract
           class can't anticipate the method name.
    """

    def __init__(self, protected_figure: Diagram, metadata: dict=None):
        self.protected_figure = protected_figure
        self.metadata = metadata or dict()

    def __enter__(self):
        if self.protected_figure.will_be_stored:
            self.protected_figure.save(metadata=self.metadata)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:  # EXITED WITH ERROR!
            if not self.protected_figure.will_be_stored:  # If it hasn't been stored already: do it now.
                self.protected_figure.save(metadata=self.metadata)

            print(f"Error writing figure '{self.protected_figure.name}'. Often that's a LaTeX issue (illegal characters somewhere?). ")
            print(f"Don't panic: your datapoints were cached, but you may have to go into the JSON to fix things.")
            print("Here's the exception, FYI:")
            print("==========================")
            time.sleep(0.5)
            import traceback
            traceback.print_exception(exc_type, exc_val, exc_tb)
            time.sleep(0.5)
            print("==========================")

        # Swallow exception
        return True


class LineGraph(Diagram):
    """
    2D line graph. Plots the relationship between TWO variables, given in paired observations (x,y), and connects them.
    For the name, see: https://trends.google.com/trends/explore?date=all&q=line%20graph,line%20chart,line%20plot
    """

    def _load(self, saved_data: dict):
        """
        Restore a file by extending the series of this object.
        """
        for name, tup in saved_data.items():
            if len(tup) != 2 or len(tup[0]) != len(tup[1]):
                raise ValueError("Graph data corrupted: either there aren't two tracks for a series, or they don't match in length.")

            self.addMany(name, tup[0], tup[1])

    def initSeries(self, series_name: str):
        self.data[series_name] = ([], [])

    def add(self, series_name: str, x, y):
        """
        Add a single datapoint to the series (line) with the given label.
        """
        if type(x) == str or type(y) == str:
            print("WARNING: You are trying to use a string as x or y data. The datapoint was discarded, because this causes nonsensical graphs.")
            return

        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].append(x)
        self.data[series_name][1].append(y)

    def addMany(self, series_name: str, xs: Sequence, ys: Sequence):
        if series_name not in self.data:
            self.initSeries(series_name)
        self.data[series_name][0].extend(xs)
        self.data[series_name][1].extend(ys)

    def commit(self, aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label="", legend_position="lower right",
               do_points=True, initial_style_idx=0,
               grid_linewidth=DEFAULT_GRIDWIDTH, curve_linewidth=1, optional_line_at_y=None,
               y_lims=None, x_tickspacing=None, y_tickspacing=None, logx=False, logy=False,
               only_for_return=False, existing_figax: tuple=None):
        """
        Render a figure based on the added data.
        Also stores the data to a JSON file (see save()).

        Since figure rendering can error due to the LaTeX compiler (for example, because your axis labels use unicode
        instead of LaTeX commands), the entire implementation is wrapped in a try-except.
        Yes, I had to find out the hard way by losing a 6-hour render.
        """
        with ProtectedData(self):
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
            for name, samples in self.data.items():
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
                               min([min(tup[0]) for tup in self.data.values()]),
                               max([max(tup[0]) for tup in self.data.values()]), colors='b', linestyles='dotted')

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
                self.exportToPdf(fig)
            return fig, main_ax

    def merge_commit(self, fig, ax1, other_graph: "LineGraph", **second_commit_kwargs):
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
        other_graph.commit(**second_commit_kwargs, existing_figax=(fig,ax2),
                           initial_style_idx=len(self.data), only_for_return=True)

        # Drawing the legends is slightly tricky, see https://stackoverflow.com/a/54631364/9352077
        legend_1 = ax1.legend(loc='upper right')
        legend_1.remove()
        ax2.legend(loc='lower right')
        ax2.add_artist(legend_1)

        # At last, save.
        Diagram.safeFigureWrite(name, ".pdf", fig)

    @staticmethod
    def qndLoadAndCommit(json_path: Path):
        """
        Quick-and-dirty method to load in the JSON data of a graph and commit it without
        axis formatting etc. Useful when rendering with commit() failed but your JSON was
        saved, and you want to get a rough idea of what came out of it.

        Strips the part of the stem at the end that starts with "_".
        """
        raw_name = json_path.stem
        g = LineGraph(raw_name[:raw_name.rfind("_")], caching=CacheMode.NONE)
        g.load(json_path)
        g.commit()


class MergedLineGraph(Diagram):
    """
    Merger of two line graphs.
    The x axis is shared, the first graph's y axis is on the left, and the second graph's y axis is on the right.
    """

    def __init__(self, g1: LineGraph, g2: LineGraph,
                 use_cached: bool):
        self.g1 = g1
        self.g2 = g2
        super().__init__(name=self.makeName(), use_cached=use_cached)

    def makeName(self):
        return self.g1.name + "+" + self.g2.name

    def _save(self) -> dict:
        return {"G1": {"name": self.g1.name,
                       "data": self.g1._save()},
                "G2": {"name": self.g2.name,
                       "data": self.g2._save()}}

    def _load(self, saved_data: dict):
        # The KeyError thrown when one of these isn't present, is sufficient.
        name, data = saved_data["G1"]["name"], saved_data["G1"]["data"]
        self.g1.name = name
        self.g1._load(data)
        name, data = saved_data["G2"]["name"], saved_data["G2"]["data"]
        self.g2.name = name
        self.g2._load(data)

        if self.name != self.makeName():
            raise ValueError("Graph names corrupted: found", self.g1.name, "and", self.g2.name, "for merge", self.name)

    def commit(self, aspect_ratio=DEFAULT_ASPECT_RATIO, x_label="", y_label_left="", y_label_right=""):
        with ProtectedData(self):
            ######## ALREADY IN .COMMIT
            # First graph
            colours = getColours()
            fig, ax1 = newFigAx(aspect_ratio)
            ax1.grid(True, which='both')
            ax1.axhline(y=0, color='k', lw=0.5)

            for name, samples in self.g1.data.items():
                ax1.plot(samples[0], samples[1], c=colours.pop(0), marker=".", linestyle="-", label=name)
            ########

            # Second graph
            ax2 = ax1.twinx()  # "Twin x" means they share the same figure and x axis, but the other's y axis will be on the right.

            ######## ALREADY IN .COMMIT
            for name, samples in self.g2.data.items():
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

            self.exportToPdf(fig)


class Bars(Diagram):
    """
    Multi-bar chart. Produces a chart with groups of bars on them: each group has the same amount of bars and the
    colours of the bars are in the same order per group.

    All the bars of the same colour are considered the same family. All the families must have the same amount of
    bars, which is equal to the amount of groups.
    """

    def _load(self, saved_data: dict):
        self.data = saved_data  # TODO: Needs more sanity checks

    def add(self, bar_slice_family: str, height: float):
        if bar_slice_family not in self.data:
            self.data[bar_slice_family] = []
        self.data[bar_slice_family].append(height)

    def addMany(self, bar_slice_family: str, heights: Sequence[float]):
        if bar_slice_family not in self.data:
            self.data[bar_slice_family] = []
        self.data[bar_slice_family].extend(heights)

    def commit(self, group_names: Sequence[str], bar_width: float, group_spacing: float, y_label: str="",
               diagonal_labels=True, aspect_ratio=DEFAULT_ASPECT_RATIO,
               y_tickspacing: float=None, log_y: bool=False):
        """
        The reason that group names are not given beforehand is because they are much like an x_label.
        Compare this to the family names, which are in the legend just as with LineGraph and MultiHistogram.
        """
        with ProtectedData(self):
            fig, main_ax = newFigAx(aspect_ratio)
            main_ax: plt.Axes

            colours = getColours()
            group_locations = None
            for i, (bar_slice_family, slice_heights) in enumerate(self.data.items()):
                group_locations = group_spacing * np.arange(len(slice_heights))
                main_ax.bar(group_locations + bar_width*i, slice_heights, color=colours.pop(0), width=bar_width,
                            label=bar_slice_family)

            # X-axis
            main_ax.set_xticks(group_locations + bar_width * len(self.data) / 2 - bar_width / 2)
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

            self.exportToPdf(fig)


class MultiHistogram(Diagram):
    """
    A histogram plots the distribution of a SINGLE variable.
    On the horizontal axis is that variable's domain. On the vertical axis is the frequency/fraction/... of each value.
    That means: you only give values of the variable, and the heights are AUTOMATICALLY computed, unlike in a graph.

    Can be seen as a cross between a graph and a bar chart.
    """

    def _load(self, saved_data: dict):
        for name, values in saved_data.items():
            if not isinstance(values, list):
                raise ValueError("Histogram data corrupted.")
            self.addMany(name, values)

    def add(self, series_name: str, x_value: float):
        if series_name not in self.data:
            self.data[series_name] = []
        self.data[series_name].append(x_value)

    def addMany(self, series_name: str, values: Sequence[float]):
        if series_name not in self.data:
            self.data[series_name] = []
        self.data[series_name].extend(values)

    def toDataframe(self):
        # You would think all it takes is pd.Dataframe(dictionary), but that's the wrong way to do it.
        # If you do it that way, you are pairing up the i'th value of each family as if it belongs to one
        # object. This is not correct, and crashes if you have different sample amounts per family.
        rows = []  # For speed, instead of appending to a dataframe, make a list of rows as dicts. https://stackoverflow.com/a/17496530/9352077
        for name, x_values in self.data.items():
            for v in x_values:
                rows.append({"value": v, LEGEND_TITLE_CLASS: name})
        df = pd.DataFrame(rows)
        return df if len(self.data) != 1 else df.drop(columns=[LEGEND_TITLE_CLASS])

    def commit(self, width: float, x_label="", y_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        with ProtectedData(self):
            fig, main_ax = newFigAx(aspect_ratio)
            for name,x_values in self.data.items():
                main_ax.hist(x_values, bins=int((max(x_values) - min(x_values)) // width))  # You could do this, but I don't think the ticks will then be easy to set.

            if x_label:
                main_ax.set_xlabel(x_label)
            if y_label:
                main_ax.set_ylabel(y_label)

            self.exportToPdf(fig)

    def commit_histplot(self, binwidth: float=1, log_x=False, log_y=False,
                        relative_counts: bool=False, average_over_bin: bool=False,
                        x_lims: Tuple[int,int]=None, aspect_ratio=DEFAULT_ASPECT_RATIO,
                        x_tickspacing: float=1, y_tickspacing: float=None, center_ticks=False,
                        do_kde=True, kde_smoothing=True,
                        border_colour=None, fill_colour=None,  # Note: colour=None means "use default colour", not "use no colour".
                        x_label: str="", y_label: str="",
                        **seaborn_args):
        """
        :param x_lims: left and right bounds. Either can be None to make them automatic. These bound are the edge of
                       the figure; if you have a bar from x=10 to x=11 and you set the right bound to x=10, then you
                       won't see the bar but you will see the x=10 tick.
        :param center_ticks: Whether to center the ticks on the bars. The bars at the minimal and maximal x_lim are
                             only half-visible.
        """
        with ProtectedData(self):
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
            if len(self.data) != 1:
                legend_title = LEGEND_TITLE_CLASS
                print(df.groupby(LEGEND_TITLE_CLASS).describe())
            else:
                legend_title = None
                print(df.describe())

            fig, ax = newFigAx(aspect_ratio)
            if not log_x:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,  # x and hue args: https://seaborn.pydata.org/tutorial/distributions.html
                             binwidth=binwidth, binrange=(math.floor(df["value"].min()/binwidth)*binwidth,
                                                          math.ceil( df["value"].max()/binwidth)*binwidth),
                             discrete=center_ticks, stat=mode, common_norm=False,
                             kde=do_kde, kde_kws={"bw_adjust": 10} if kde_smoothing else None,  # Btw, do not use displot: https://stackoverflow.com/a/63895570/9352077
                             color=fill_colour, edgecolor=border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077
            else:
                sns.histplot(df, ax=ax, x="value", hue=legend_title,
                             log_scale=True,
                             discrete=center_ticks, stat=mode, common_norm=False,
                             color=fill_colour, edgecolor=border_colour,
                             **seaborn_args)  # Do note use displot: https://stackoverflow.com/a/63895570/9352077

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label + r" [\%]" * (mode == "percent" and y_label != ""))
            if x_lims:
                if x_lims[0] is not None and x_lims[1] is not None:
                    ax.set_xlim(x_lims[0], x_lims[1])
                elif x_lims[0] is not None:
                    ax.set_xlim(left=x_lims[0])
                elif x_lims[1] is not None:
                    ax.set_xlim(right=x_lims[1])
                else:  # You passed (None,None)...
                    pass

            # Weird tick spacing hack that somehow works https://stackoverflow.com/a/44525175/9352077
            if not log_x:
                ax.xaxis.set_major_locator(tkr.MultipleLocator(x_tickspacing))
                ax.xaxis.set_major_formatter(tkr.ScalarFormatter())

            if not log_y:
                if y_tickspacing:
                    ax.yaxis.set_major_locator(tkr.MultipleLocator(y_tickspacing))
                    ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
            else:
                ax.set_yscale("log")

            # ax.set_yticklabels(np.arange(0, max(self.x_values), ytickspacing, dtype=int))  # Don't do this. It literally overwrites existing ticks, rather than placing more of them, so the result is mathematically wrong.
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linewidth=DEFAULT_GRIDWIDTH)
            self.exportToPdf(fig, stem_suffix="_histplot")

    def commit_boxplot(self, value_axis_label: str= "", class_axis_label: str= "",
                       aspect_ratio=DEFAULT_ASPECT_RATIO,
                       log=False, horizontal=False, iqr_limit=1.5,
                       value_tickspacing=None):
        """
        Draws multiple boxplots side-by-side.
        Note: the "log" option doesn't just stretch the value axis, because in
        that case you will get a boxplot on skewed data and then stretch that bad
        boxplot. Instead, this method applies log10 to the values, then computes
        the boxplot, and plots it on a regular axis.
        """
        with ProtectedData(self):
            rows = []
            for name, x_values in self.data.items():
                for v in x_values:
                    if log:
                        rows.append({"value": np.log10(v), LEGEND_TITLE_CLASS: name})
                    else:
                        rows.append({"value": v, LEGEND_TITLE_CLASS: name})
            df = pd.DataFrame(rows)
            print(df.groupby(LEGEND_TITLE_CLASS).describe())

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
                sns.boxplot(df, x="value", y=LEGEND_TITLE_CLASS,
                            ax=ax, linewidth=0.5, flierprops=flierprops)
                ax.set_xlabel(value_axis_label)
                ax.set_ylabel(class_axis_label)
            else:
                sns.boxplot(df, x=LEGEND_TITLE_CLASS, y="value",
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
            self.exportToPdf(fig, stem_suffix="_boxplot")


class Histogram(MultiHistogram):
    """
    Simplified interface for a histogram of only one variable.
    """

    def _load(self, saved_data: dict):
        super()._load(saved_data)
        if not (len(self.data) == 1 and "x_values" in self.data):
            raise ValueError("Histogram data corrupted.")

    def add(self, x_value: float):
        super().add("x_values", x_value)

    def addMany(self, values: Sequence[float]):
        super().addMany("x_values", values)

    def commit_boxplot(self, x_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.boxplot(df, x="value", showmeans=True, orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([""])
        self.exportToPdf(fig, stem_suffix="_boxplot")

    def commit_violin(self, x_label="", y_label="", aspect_ratio=DEFAULT_ASPECT_RATIO):
        fig, ax = newFigAx(aspect_ratio)
        df = self.toDataframe()

        sns.violinplot(df, x="value", orient="h", ax=ax)

        ax.set_xlabel(x_label)
        ax.set_yticklabels([y_label], rotation=90, va='center')
        self.exportToPdf(fig, stem_suffix="_violinplot")


class ScatterPlot(Diagram):

    def _load(self, saved_data: dict):
        self.data = saved_data  # FIXME: Needs more sanity checks obviously

    def copy(self, new_name: str):
        new_plot = ScatterPlot(new_name, use_cached=False)
        for name, values in self.data.items():
            new_plot.addPointsToFamily(name, values[0].copy(), values[1].copy())
        return new_plot

    def addPointsToFamily(self, family_name: str, xs, ys):
        """
        Unlike the other diagram types, it seems justified to add scatterplot points in bulk.
        Neither axis is likely to represent time, so you'll probably have many points available at once.
        """
        if family_name not in self.data:
            self.data[family_name] = ([], [])
        self.data[family_name][0].extend(xs)
        self.data[family_name][1].extend(ys)

    def commit(self, aspect_ratio: Tuple[float,float]=DEFAULT_ASPECT_RATIO, x_label="", y_label="",
               x_lims=None, y_lims=None, logx=False, logy=False, x_tickspacing=None, grid=False, legend=False,
               family_colours=None, family_sizes=None, randomise_markers=False, only_for_return=False):
        with ProtectedData(self):
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
            cols = plt.cm.rainbow(np.linspace(0, 1, len(self.data)))  # Equally spaced rainbow colours.
            scatters = []
            names    = []
            for idx, tup in enumerate(sorted(self.data.items(), reverse=True)):
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

            if not only_for_return:
                self.exportToPdf(fig)
            return fig, ax


class Table(Diagram):
    """
    Structure with named rows and infinitely many nested named columns, AND with an order for all columns and all rows.
    A good example of this kind of table is the morphemic-lexemic unweighted-weighted Pr-Re-F1 tables in my thesis.
    """

    @dataclass
    class Column:
        name: str
        subcolumns: List["Table.Column"]
        rows: Dict[str,float]

        @staticmethod
        def fromDict(asdict: dict):  # inverse of dataclasses.asdict
            assert "name" in asdict and "subcolumns" in asdict and "rows" in asdict
            assert isinstance(asdict["subcolumns"], list)
            return Table.Column(asdict["name"], [Table.Column.fromDict(d) for d in asdict["subcolumns"]], asdict["rows"])

    def set(self, value: float, row: str, column_path: List[str]):
        """
        You could choose to either store all columns for all rows, or store all rows for all columns.
        The former is more intuitive when indexing (you start with the row), but actually, it makes way more sense to
        only store the structure of the table once.
        """
        if not self.data:
            self.data["top-column"] = Table.Column("", [], dict())
            self.data["unique-rows"] = []

        current_col: Table.Column = self.data["top-column"]
        for col in column_path:
            current_list = current_col.subcolumns
            for i in range(len(current_list)):
                if current_list[i].name == col:
                    current_col = current_list[i]
                    break
            else:
                current_list.append(Table.Column(col, [], dict()))
                current_col = current_list[-1]

        current_col.rows[row] = value
        if row not in self.data["unique-rows"]:
            self.data["unique-rows"].append(row)

    def _save(self) -> dict:
        return {
            "top-column": dataclasses.asdict(self.data["top-column"]),
            "unique-rows": self.data["unique-rows"]
        }

    def _load(self, saved_data: dict):
        self.data = {
            "top-column": Table.Column.fromDict(saved_data["top-column"]),
            "unique-rows": saved_data["unique-rows"]
        }

    # TODO: Needs a .commit method obviously. Probably a good idea to commit to .tex.


def arrow(ax: plt.Axes, start_point, end_point):  # FIXME: I want TikZ's stealth arrows, but this only seems possible in Matplotlib's legacy .arrow() interface (which doesn't keep its head shape properly): https://stackoverflow.com/a/43379608/9352077
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
