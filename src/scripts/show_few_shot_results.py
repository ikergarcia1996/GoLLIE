import glob
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SUPERVISED_TASKS = [
    "ace05.eae",
    "ace05.ner",
    "ace05.rc",
    "ace05.ver",
    "bc5cdr.ner",
    "conll03.ner",
    "diann.ner",
    "ncbidisease.ner",
    "ontonotes5.ner",
    "rams.eae",
    "tacred.sf",
    "wnut17.ner",
]


def main(output_dir: str, metric: str = "f1-score", result_path: str = "assets/results", drop_gold: bool = True):
    results = defaultdict(dict)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Load results
    for chkpt_dir in glob.glob(f"{output_dir}/checkpoint-*"):
        chkpt = int(chkpt_dir.split("/")[-1].split("-")[-1])

        try:
            with open(f"{chkpt_dir}/task_scores_summary.json", "rt") as f:
                task_scores = json.load(f)

            results[chkpt] = task_scores
        except FileNotFoundError:
            print(f"Skipping {chkpt_dir}")

    results_df = []
    for chkpt, task_scores in results.items():
        for task, k_shots in task_scores.items():
            for k_shot, info in k_shots.items():
                for subtask, metrics in info.items():
                    for _metric, value in metrics.items():
                        print(_metric)
                        if _metric == metric:
                            print(value)
                            results_df.append(
                                {
                                    "checkpoint": chkpt,
                                    "task": task,
                                    "subtask": subtask,
                                    "supervised": task in SUPERVISED_TASKS,
                                    "k_shot": int(k_shot.split("-")[-1]),
                                    # "metric": _metric,
                                    "value": value * 100,
                                }
                            )

    results_df = pd.DataFrame(results_df)
    if drop_gold:
        print("Dropping gold results")
        results_df = results_df[
            ((results_df["task"] != "tacred.sf") | (results_df["subtask"] != "templates"))
        ].reset_index(drop=True)
        results_df = results_df[
            ((results_df["task"] != "rams.eae") | (results_df["subtask"] != "events"))
        ].reset_index(drop=True)
    # print(results_df.head(10))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(data=results_df, x="checkpoint", y="value", hue="supervised", style="k_shot", ax=ax, errorbar=None)
    plt.grid(True)

    fig.savefig(f"{result_path}/{os.path.basename(output_dir.rstrip('/'))}_few_shot_results.png", dpi=300)

    compact_results = None
    for k in sorted(results_df["k_shot"].unique()):
        k_results = (
            results_df[results_df["k_shot"] == k]
            .sort_values(["checkpoint", "supervised", "task", "subtask"], ascending=True)
            .reset_index(drop=True)
        )
        if compact_results is None:
            compact_results = k_results
            compact_results.drop(columns=["k_shot"], inplace=True)
            compact_results.rename(columns={"value": f"value_{k}"}, inplace=True)
        else:
            compact_results[f"value_{k}"] = k_results["value"]

    print(compact_results[compact_results["checkpoint"] == 419].drop(columns=["checkpoint"]).to_markdown(floatfmt=".2f"))
    print("Average:")
    print(results_df.groupby(["checkpoint", "supervised", "k_shot"])["value"].mean())
    return

    print("Results")
    print(
        results_df[(results_df["k_shot"] == 0) & (results_df["checkpoint"] == 279)].sort_values(
            ["supervised", "task", "subtask"], ascending=True
        )
    )
    print("Average:")
    print(results_df[(results_df["checkpoint"] == 279)].groupby(["supervised", "k_shot"])["value"].mean())


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True, help="Path to the results directory")
    parser.add_argument("--metric", type=str, default="f1-score", help="Metric to use for sorting")
    parser.add_argument("--result_path", type=str, default="assets/results", help="Path to the results directory")
    parser.add_argument(
        "--include_gold", action="store_false", dest="drop_gold", default=True, help="Drop gold results"
    )

    main(**vars(parser.parse_args()))
