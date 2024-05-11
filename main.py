from pathlib import Path

import hydra
import omegaconf
import wandb
from loguru import logger
from pytorch_lightning import seed_everything
from ttimer import get_timer

from src import Config, create_optimizer, evaluate_pi, exam_func, synthesize_rel_mat


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: Config) -> None:
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    project_name = "nsw-with-optimal-transport"
    current_dir_name = Path.cwd().name
    exp_run = f"{cfg.exp_name}_{current_dir_name}"
    wandb.init(project=project_name, name=exp_run, config=wandb_config)  # type: ignore

    # generate data
    seed_everything(cfg.seed)
    generator_cfg = cfg.generator
    rel_mat_true, rel_mat_obs = synthesize_rel_mat(
        generator_cfg.n_query,
        generator_cfg.n_doc,
        generator_cfg.lam,
        generator_cfg.flip_ratio,
        generator_cfg.noise,
        cfg.seed,
    )
    expo = exam_func(generator_cfg.K, generator_cfg.shape)

    # solve
    optimizer = create_optimizer(cfg.optimizer.name, **cfg.optimizer.params)
    timer = get_timer(timer_name="optimization")
    with timer(name="solve"):
        pi = optimizer.solve(rel_mat_obs, expo)
    exec_time = timer["solve"].time
    logger.info(f"\n{timer.render()}")

    # evaluate
    result = evaluate_pi(pi, rel_mat_true, expo)
    result["exec_time"] = exec_time
    logger.info(result)

    # log
    wandb.log(result)
    wandb.finish()


if __name__ == "__main__":
    main()
