# Fast solution to the fair ranking problem using the Sinkhorn algorithm

This repository contains the code used for the experiments in "Fast solution to the fair ranking problem using the Sinkhorn algorithm" (PRICAI 2024)

Note: This paper is under review.

## Abstract
In two-sided marketplaces such as online flea markets, recommender systems for providing consumers with personalized item rankings play a key role in promoting transactions between providers and consumers. 
Meanwhile, two-sided marketplaces face the problem of balancing consumer satisfaction and fairness among items to stimulate activity of item providers.
Saito and Joachims (2022) devised an impact-based fair ranking method for maximizing the Nash social welfare based on fair division; however, this method, which requires solving a large-scale constrained nonlinear optimization problem, is very difficult to apply to practical-scale recommender systems.
We thus propose a fast solution to the impact-based fair ranking problem. 
We first transform the fair ranking problem into an unconstrained optimization problem and then design a gradient ascent method that repeatedly executes the Sinkhorn algorithm. 
Experimental results demonstrate that our algorithm provides fair rankings of high quality and is about 1000 times faster than application of commercial optimization software.

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
dependencies = [
    ...,
    "torch == 2.3.0+cu121", # change this to match your environment
]

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

```python
from src import create_generator, create_optimizer, evaluate_pi

g = create_generator(
    generator_name="synthetic",
    # specific generator parameters
    n_query=250,
    n_doc=1600,
    lam=1.0,
)
rel_mat_true, rel_mat_obs = g.generate_rel_mat()  # relevance matrix
expo = g.exam_func(K=10, shape="inv")  # exposure

optimizer = create_optimizer(
    optimizer_name="ot_nsw",
    # specific optimizer parameters
    eps=1,
    lr=0.5,
    use_amp=False,
    tol=0.01,
    device="cuda",
)
pi = optimizer.solve(rel_mat_obs, expo)

print(evaluate_pi(pi, rel_mat_true, expo))
```

## List available optimizers and generators
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
