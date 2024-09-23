import glob
import json
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


def main(output_dir: str, metric: str = "f1-score", result_path: str = "assets/results"):
    results = defaultdict(dict)

    # Load results
    for chkpt_dir in glob.glob(f"{output_dir}/checkpoint-*"):
        chkpt = int(chkpt_dir.split("/")[-1].split("-")[-1])

        with open(f"{chkpt_dir}/task_scores_summary.json", "rt") as f:
            task_scores = json.load(f)

        results[chkpt] = task_scores

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
                                    "value": value,
                                }
                            )

    results_df = pd.DataFrame(results_df)
    print(results_df.head(10))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(data=results_df, x="checkpoint", y="value", hue="supervised", style="k_shot", ax=ax, errorbar=None)
    plt.grid(True)

    fig.savefig(f"{result_path}/few_shot_results.png", dpi=300)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True, help="Path to the results directory")
    parser.add_argument("--metric", type=str, default="f1-score", help="Metric to use for sorting")
    parser.add_argument("--result_path", type=str, default="assets/results", help="Path to the results directory")

    main(**vars(parser.parse_args()))
