from pathlib import Path

import click
import pandas as pd
import wandb


# 再帰的にdictを展開する
def flatten_dict(d: dict, parent_key: str = "", sep: str = "_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@click.command()
@click.option("--user_name", type=str, default="213tubo", help="wandb user name")
@click.option("--output_dir", type=Path, default="./output", help="output directory")
def export_result(user_name: str, output_dir: Path):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{user_name}/nsw-with-optimal-transport")

    data_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        data = flatten_dict(run.summary._json_dict)

        # .config contains the hyperparameters.
        # run.configをflattenする
        config = flatten_dict(run.config)
        for k, v in config.items():
            data[k] = v

        # .name is the human-readable name of the run.
        data["name"] = run.name

        data_list.append(data)

    runs_df = pd.DataFrame(data_list)
    output_dir.mkdir(exist_ok=True, parents=True)
    runs_df.to_csv(output_dir / "result.csv", index=False)


if __name__ == "__main__":
    export_result()
