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

#### Set the dependency resolution method to [uv](https://astral.sh/blog/uv)
```bash
rye config --set-bool behavior.use-uv=true
```

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

## Datasets

(This explanation is based on the repository of the [kdd2022-fair-ranking-nsw](https://github.com/usaito/kdd2022-fair-ranking-nsw/tree/main) by yuta-saito)

We use "Delicious" and "Wiki10-31K" from The Extreme Classification Repository. Please install the above datasets from the repository and put them under the ./data/ as follows.

```
root/
├── data
│   ├── delicious.txt
│   └── wiki.txt
```

Note that we rename the files as follows:
- Delicious_data.txt in Delicious.zip to delicious.txt
- train.txt in Wiki10.bow.zip to wiki.txt , respectively.

## Basic Usage

### Generate Data
```python
from src import create_generator

g = create_generator(generator_name='synthetic', n_query=100, n_doc=1000)
rel_mat_true, rel_mat_obs = g.generate_rel_mat() # relevance matrix
expo = g.exam_func(K=10) # exposure
```

### Optimize
```python
from src import create_optimizer, evaluate_pi

optimizer = create_optimizer(optimizer_name='ot_nsw')
pi = optimizer.solve(rel_mat_obs, expo)

print(evaluate_pi(pi, rel_mat_true, expo))
```

### List of available optimizers and generators
```python
from src import list_optimizers, list_generators

print(list_optimizers())
print(list_generators())
```


## Run Experiment

Before running an experiment, you need register wandb.
please see [wandb documentation](https://docs.wandb.ai/ja/quickstart) for details.


You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with optimizer=nsw, number of documents of 100, 200, 300, and 400.

```bash
rye run python main.py -m optimizer=nsw generator.n_doc=100,200,300,400
```

### Reproduce the results of the paper

```bash
make run-all
```

## Export experiment results

The following command exports the experiment results from wandb to the local directory.

```bash
rye run python tools/export_results.py --user_name {your wandb user name}
```

# References

- Saito, Y., & Joachims, T. (2022, August). Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 1514-1524).
