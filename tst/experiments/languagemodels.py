"""
Formats (not generates) the CSVs obtained from Weights & Biases for the LM training.
"""
from tst.preamble import *

from typing import Tuple
import csv
import re

from fiject import LineGraph, CacheMode, Table, ColumnStyle


PATH_PRETRAINING_RESULTS = PATH_DATA_OUT / "nl-mlm-pretraining.csv"
PATH_FINETUNING_RESULTS  = PATH_DATA_OUT / "nl-mlm-finetuning.csv"

BPE_NAME   = "BPE"
KNOCK_NAME = "BPE-knockout"
CONV_NAME  = r"BPE $\to$ BPE-knockout"
def task_formatter(wandb_name: str) -> Tuple[str, str]:
    if "sa" in wandb_name:
        task_formatted = "SA"
        task_family = "Sequence-level"
    elif "nli" in wandb_name:
        task_formatted = "NLI"
        task_family = "Sequence-level"
    elif "pos" in wandb_name:
        task_formatted = "PoS"
        task_family = "Token-level"
    elif "ner" in wandb_name:
        task_formatted = "NER"
        task_family = "Token-level"
    elif "pppl" in wandb_name:
        task_formatted = "PPPL"
        task_family = ""
    else:
        raise RuntimeError()
    return task_formatted, task_family


def family_formatter(wandb_name: str) -> str:
    if "conversion" in wandb_name:
        return CONV_NAME
    elif "knockout" in wandb_name:
        return KNOCK_NAME
    else:
        return BPE_NAME


def main_finetuningTable():
    keys = ["Group", "global_step", "trainer/global_step", "ner-f1", "nli-acc", "pos-f1", "sa-acc", "pppl"]
    task_order = ["pppl", "sa-acc", "nli-acc", "ner-f1", "pos-f1"]

    relevant = []
    with open(PATH_FINETUNING_RESULTS, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if any(row[k] != "" for k in task_order):
                relevant.append({k:v for k,v in row.items() if k in keys and v != ""})

    tasks = dict()
    for row in relevant:
        step, family = row.pop("global_step"), row.pop("Group")
        task, result = row.popitem()
        family = family_formatter(family)

        if task not in tasks:
            tasks[task] = dict()
        if family not in tasks[task]:
            tasks[task][family] = []

        tasks[task][family].append((int(step), float(result)))
        tasks[task][family].sort()

    # for task in tasks:
    #     graph = LineGraph("extrinsic-nl-" + task, caching=CacheMode.NONE)
    #     for family in tasks[task]:
    #         for x,y in tasks[task][family]:
    #             graph.add(family, x, y)
    #     graph.commitWithArgs(LineGraph.ArgsGlobal(y_label=task, y_lims=(0.50,1.01), y_tickspacing=0.1), LineGraph.ArgsPerLine())

    family_order = [BPE_NAME, KNOCK_NAME, CONV_NAME]
    unique_iterations = set()
    table = Table("extrinsic-nl", caching=CacheMode.NONE)
    for task in sorted(tasks, key=task_order.index):
        task_formatted, task_family = task_formatter(task)

        for family in sorted(tasks[task], key=family_order.index):
            for x,y in tasks[task][family]:
                x = round_to_nearest_1000(x)
                if x < 30000:
                    continue
                x_formatted = str(x // 1000) + "k"
                unique_iterations.add(x_formatted)

                # row_identifier = ["Dutch", family_formatted, r"$\vphantom{" + str(x) + "}$"]
                # table.set(x, row_identifier, ["Pre-training", "Iter.\\"])
                # table.set(0, row_identifier, ["Pre-training", "PPPL"])
                # table.set(y, row_identifier, ["Fine-tuning", task_family, task_formatted])
                row_identifier = ["Dutch", family]
                if task_formatted == "PPPL":
                    table.set(y, row_identifier, ["PPPL", x_formatted])
                else:
                    table.set(y, row_identifier, [task_family, task_formatted, x_formatted])

    table.commit(borders_between_columns_of_level=[0,1], borders_between_rows_of_level=[1],
                 default_column_style=ColumnStyle(aggregate_at_rowlevel=0, do_bold_maximum=True,
                                                  cell_prefix=r"\tgrad[0][50][100]{", cell_function=lambda x: 100*x, cell_suffix="}", cell_default_if_empty=r"\cellcolor{black!20}"),
                 alternate_column_styles={
                     ("Pre-training", "Iter.\\"): ColumnStyle(cell_prefix=r"$\num{", cell_suffix="}$", digits=0),
                     ("Pre-training", "PPPL"): ColumnStyle(do_bold_minimum=True, cell_default_if_empty=r"\cellcolor{black!20}")
                 } | {
                     ("PPPL", x): ColumnStyle(do_bold_minimum=True, cell_default_if_empty=r"\cellcolor{black!20}")
                     for x in sorted(unique_iterations)
                 })


def round_to_nearest_1000(x: int):
    return 1000*round(x/1000)


def main_pretrainingGraph(aspect_ratio=(5,3.25)):
    NAME_PATTERN = re.compile(r"Group: (.+) - validation/loss_epoch")
    # keys = ["Group", "global_step", "trainer/global_step", "ner-f1", "nli-acc", "pos-f1", "sa-acc"]
    # nonempty = ["ner-f1", "nli-acc", "pos-f1", "sa-acc"]

    suffices = ['global_step', 'validation/loss_epoch']

    relevant = []
    with open(PATH_PRETRAINING_RESULTS, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relevant.append({k:v for k,v in row.items() if any(k.endswith(s) for s in suffices) and v != ""})

    graph = LineGraph("extrinsic-nl-mlm", CacheMode.NONE)
    for row in relevant:
        # Get x
        x = None
        for key in row:
            if "step" in key:
                x = int(row[key])
        assert x is not None

        # Get y
        for key in row:
            m = NAME_PATTERN.match(key)
            if m:
                family = family_formatter(m.group(1))
                if (family == BPE_NAME   and x > 39_000) or \
                   (family == KNOCK_NAME and x > 30_000) or \
                   (family == CONV_NAME  and x > 39_000):
                    continue
                graph.add(family, x, float(row[key]))

    graph.commitWithArgs(
        LineGraph.ArgsGlobal(x_label="Training batches", y_label="Validation loss", legend_position="upper right", logy=True,
                             aspect_ratio=aspect_ratio, y_tickspacing=1, tick_scientific_notation=False, tick_log_multiples=True,
                             curve_linewidth=1.5),
        LineGraph.ArgsPerLine(show_points=False)
    )


if __name__ == "__main__":
    from fiject import FIJECT_DEFAULTS
    FIJECT_DEFAULTS.RENDERING_FORMAT = "pdf"

    UPSCALE = 1.9
    main_pretrainingGraph((2.25*UPSCALE,1*UPSCALE))
