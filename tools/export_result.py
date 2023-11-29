from pathlib import Path

import click
import pandas as pd
import wandb


@click.command()
@click.option("--user_name", type=str, default="213tubo", help="wandb user name")
@click.option("--output_dir", type=Path, default="./output", help="output directory")
def export_result(user_name: str, output_dir: Path):
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{user_name}/nsw-with-optimal-transport")

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True, parents=True)
    runs_df.to_csv(output_dir / "result.csv")

if __name__ == "__main__":
    export_result()
