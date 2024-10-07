import copy
import json
import logging
import multiprocessing as mp
import os
import random
from argparse import ArgumentParser
from functools import partial
from itertools import cycle
from typing import Type

import datasets
from tqdm import tqdm


def get_class(class_path: str) -> Type:
    components = class_path.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod


def shuffle_conversations(conversations):
    conversation_pairs = [(conversations[i * 2], conversations[(i * 2) + 1]) for i in range(len(conversations) // 2)]
    random.shuffle(conversation_pairs)

    return [item for sublist in conversation_pairs for item in sublist]


def multicpu_generator(args, tqdm_position, config):
    dataloader_cls = get_class(config["dataloader_cls"])
    sampler_cls = get_class(config["sampler_cls"])
    seeds = config.get("seed", 0)
    if isinstance(seeds, int):
        seeds = [seeds]
    label_noise = config.get("label_noise_prob", 0.0)
    if isinstance(label_noise, float):
        label_noise = cycle([label_noise])

    if "train_file" in config:
        dataloader = dataloader_cls(config["train_file"], **config)
        for ie_task in config["tasks"]:
            for seed, noise_prob in zip(seeds, label_noise):
                config["seed"] = seed
                config["label_noise_prob"] = noise_prob
                # Avoid multiple values for keyword argument
                _kwargs = {**config, **config["task_configuration"][ie_task]}
                sampler = sampler_cls(
                    dataloader,
                    task=ie_task,
                    split="train",
                    **_kwargs,
                )

                output_name = f"{config['dataset_name'].lower()}.{ie_task.lower()}.train.{seed}.jsonl"

                if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
                    logging.warning(f"Skipping {output_name} because it already exists.")
                    continue

                with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
                    total=len(dataloader),
                    desc=f"{config['dataset_name']}-{ie_task}-train-{seed}",
                    position=tqdm_position,
                ) as progress:
                    ids = []
                    for elem in sampler:
                        _file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
                        if ids != elem["ids"]:
                            ids = elem["ids"]
                            progress.update(len(ids))

                logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")

    if (
        "few-shot_examples_from" in config
        and max(args.k_shots) > 0
        and config["few-shot_examples_from"]
        and ("dev_file" in config or "test_file" in config)
    ):
        if not config["use_chat_format"]:
            raise NotImplementedError("Few-shot examples are only supported for chat format.")

        few_shot_examples = {}
        if config["few-shot_examples_from"].startswith("prepro:"):
            raise NotImplementedError("Few-shot examples from JSONL files are not supported yet.")
        elif config["few-shot_examples_from"] == "self":
            logging.info("Few-shot examples are generated from the file itself.")
        else:
            config["seed"] = 0
            logging.warning(config.__str__())
            few_shot_dataloader = dataloader_cls(config["few-shot_examples_from"], **config)
            for task in config["tasks"]:
                _kwargs = {**config, **config["task_configuration"][task]}
                k = max(args.k_shots)
                _kwargs["parallel_instances"] = k

                sampler = sampler_cls(
                    few_shot_dataloader,
                    task=task,
                    split="dev",
                    **_kwargs,
                )
                # examples = next(iter(sampler))["conversation"][1:]
                examples = [example["conversation"][1:] for example in sampler]
                few_shot_examples[task] = examples
                # logging.warning(f"{k} Few-shot examples for {task}: {examples}")
                logging.warning(f"{len(examples)} {k}-shot example batches for {task}.")
    else:
        few_shot_examples = None

    if "dev_file" in config:
        config["seed"] = 0
        dataloader = dataloader_cls(config["dev_file"], **config)
        for task in config["tasks"]:
            _kwargs = {**config, **config["task_configuration"][task]}

            # Handle few-shot examples
            k_shots = [0] + args.k_shots
            # k_shots = [k + 1 for k in k_shots]
            for k in k_shots:
                _kwargs["parallel_instances"] = 1

                _task_few_shots = few_shot_examples[task] if few_shot_examples else None
                if _task_few_shots:
                    _task_few_shots = [batch[: k * 2] for batch in _task_few_shots]

                sampler = sampler_cls(
                    dataloader,
                    task=task,
                    split="dev",
                    few_shot_examples=_task_few_shots,
                    **_kwargs,
                )

                if k > 1:
                    output_name = f"{config['dataset_name'].lower()}.{task.lower()}.k-{k}.dev.jsonl"
                else:
                    output_name = f"{config['dataset_name'].lower()}.{task.lower()}.dev.jsonl"

                if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
                    logging.warning(f"Skipping {output_name} because it already exists.")
                    continue

                with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
                    total=len(dataloader),
                    desc=f"{config['dataset_name']}-{task}-dev",
                    position=tqdm_position,
                ) as progress:
                    ids = []
                    for elem in sampler:
                        _file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
                        if ids != elem["ids"]:
                            ids = elem["ids"]
                            progress.update(len(ids))

                logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")

    if "test_file" in config:
        config["seed"] = 0
        dataloader = dataloader_cls(config["test_file"], **config)
        for task in config["tasks"]:
            _kwargs = {**config, **config["task_configuration"][task]}

            # Handle few-shot examples
            k_shots = [0] + args.k_shots
            # k_shots = [k + 1 for k in k_shots]
            for k in k_shots:
                _kwargs["parallel_instances"] = 1

                _task_few_shots = few_shot_examples[task] if few_shot_examples else None
                if _task_few_shots:
                    _task_few_shots = [batch[: k * 2] for batch in _task_few_shots]

                sampler = sampler_cls(
                    dataloader,
                    task=task,
                    split="test",
                    few_shot_examples=_task_few_shots,
                    **_kwargs,
                )

                if k > 1:
                    output_name = f"{config['dataset_name'].lower()}.{task.lower()}.k-{k}.test.jsonl"
                else:
                    output_name = f"{config['dataset_name'].lower()}.{task.lower()}.test.jsonl"

                if os.path.exists(os.path.join(args.output_dir, output_name)) and not args.overwrite_output_dir:
                    logging.warning(f"Skipping {output_name} because it already exists.")
                    continue

                with open(os.path.join(args.output_dir, output_name), "w") as _file, tqdm(
                    total=len(dataloader),
                    desc=f"{config['dataset_name']}-{task}-test",
                    position=tqdm_position,
                ) as progress:
                    ids = []
                    for elem in sampler:
                        _file.write(f"{json.dumps(elem, ensure_ascii=False)}\n")
                        if ids != elem["ids"]:
                            ids = elem["ids"]
                            progress.update(len(ids))

                logging.info(f"Data saved to {os.path.abspath(os.path.join(args.output_dir, output_name))}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    config_files = args.configs
    # We generate a new config for each train split and task, so we also parallelize over each split and task
    configs = []
    splits = ["train_file", "dev_file", "test_file"]
    data_info = {}
    for config_file in config_files:
        with open(config_file, "rt") as f:
            config = json.load(f)

        # Do not shuffle the data if set
        config["do_not_shuffle"] = args.do_not_shuffle

        # Disable paraphrases if set
        config["disable_paraphrases"] = args.disable_paraphrases

        # Check few-shot examples from
        if "few-shot_examples_from" not in config:
            config["few-shot_examples_from"] = None
            # TODO: Implement few-shot examples from train or other files

        # Remove guidelines if baseline
        config["remove_guidelines"] = args.baseline
        config["include_examples_prob"] = float(args.include_examples)
        if args.remove_masking:
            config["label_noise_prob"] = 0.0

        if args.remove_dropout:
            for task in config["tasks"]:
                config["task_configuration"][task]["guideline_dropout"] = 0.0

        # We generate a new config for each train split and task
        tasks = config["tasks"]
        for split in splits:
            for task in tasks:
                new_config = copy.deepcopy(config)
                if split in new_config:
                    for other_split in splits:
                        if other_split != split:
                            if other_split in new_config:
                                new_config.pop(other_split)
                    new_config["tasks"] = [task]

                    if config["dataset_name"] not in data_info:
                        data_info[config["dataset_name"]] = {}

                    if task not in data_info[config["dataset_name"]]:
                        data_info[config["dataset_name"]][task] = {
                            "train_file": False,
                            "dev_file": False,
                            "test_file": False,
                        }
                    data_info[config["dataset_name"]][task][split] = True
                    configs.append(new_config)

    generator_fn = partial(
        multicpu_generator,
        args,
    )

    logging.warning(f"We will generate the following data: {json.dumps(data_info, indent=4)})")

    with mp.Pool(processes=min(os.cpu_count(), len(configs))) as pool:
        pool.starmap(generator_fn, enumerate(configs))

    logging.warning(f"Data saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    datasets.logging.set_verbosity_error()
    datasets.logging.disable_progress_bar()
    parser = ArgumentParser("generate_data", description="Generate Code formatted data.")

    parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        dest="configs",
        type=str,
        help="The list of configuration files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        dest="output_dir",
        default="data/processed",
        help="Output directory where files will be saved.",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Whether to overwrite the output dir.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Whether to generate baseline data.",
    )
    parser.add_argument(
        "--include_examples",
        action="store_true",
        default=False,
        help="Whether to include examples in the data.",
    )
    parser.add_argument(
        "--remove_dropout",
        action="store_true",
        default=False,
        help="Remove guideline dropout for the ablation analysis.",
    )
    parser.add_argument(
        "--remove_masking",
        action="store_true",
        default=False,
        help="Remove guideline masking for the ablation analysis.",
    )

    parser.add_argument(
        "--do_not_shuffle",
        action="store_true",
        default=False,
        help="Do not shuffle the data.",
    )

    parser.add_argument(
        "--disable_paraphrases",
        action="store_true",
        default=False,
        help="Disable paraphrases",
    )

    parser.add_argument(
        "--eval_k-shots",
        nargs="+",
        type=int,
        default=[],
        dest="k_shots",
        help="Add K-shot evaluations.",
    )

    args = parser.parse_args()
    main(args)
