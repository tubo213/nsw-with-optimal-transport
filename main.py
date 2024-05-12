import time
from pathlib import Path

import hydra
import omegaconf
import wandb
from loguru import logger
from pytorch_lightning import seed_everything

from src import Config, create_generator, create_optimizer, evaluate_pi


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
    generator = create_generator(generator_cfg.name, **generator_cfg.params)
    rel_mat_true, rel_mat_obs = generator.generate_rel_mat()
    expo = generator.exam_func(generator_cfg.K, generator_cfg.shape)
    logger.info(f"n_query: {rel_mat_obs.shape[0]}, n_doc: {rel_mat_obs.shape[1]}")
    logger.info(f"n_rank: {expo.shape[0]}")

    # solve
    optimizer = create_optimizer(cfg.optimizer.name, **cfg.optimizer.params)
    t0 = time.perf_counter()
    pi = optimizer.solve(rel_mat_obs, expo)
    exec_time = time.perf_counter() - t0
    logger.info(f"exec_time: {exec_time:.2f} sec")

    # evaluate
    result = evaluate_pi(pi, rel_mat_true, expo)
    result["exec_time"] = exec_time
    logger.info(result)

    # log
    wandb.log(result)
    wandb.finish()


if __name__ == "__main__":
    main()
