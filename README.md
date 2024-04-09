# NSW with Optimal Transport

## Build Environment

### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

First, Please match the index url to the environment at hand to use torch.

```toml
# pyproject.toml
[[tool.rye.sources]]
name = "torch"
url = "https://download.pytorch.org/whl/cu121" # change this to match your environment
type = "index"
```

Second, create virtual environment.
```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

## Run Experiment

Before running an experiment, you need register wandb.
please see [wandb documentation](https://docs.wandb.ai/ja/quickstart) for details.


You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with optimizer=nsw, number of documents of 100, 200, 300, and 400.

```bash
rye run python main.py -m optimizer=nsw generator.n_doc=100,200,300,400
```

## Export experiment results

The following command exports the experiment results from wandb to the local directory.

```bash
rye run python export_results.py --user_name {your wandb user name}
```

# References

- Saito, Y., & Joachims, T. (2022, August). Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 1514-1524).