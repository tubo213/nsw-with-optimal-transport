from time import time

import hydra
import omegaconf
import wandb
from pytorch_lightning import seed_everything

from src.conf import Config
from src.evaluator import evaluate_pi
from src.generator import exam_func, synthesize_rel_mat
from src.optimizer.common import get_optimizer


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def main(cfg: Config):
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="nsw-with-optimal-transport", name=cfg.exp_name, config=wandb_config)  # type: ignore

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

    optimizer = get_optimizer(cfg.optimizer.name, cfg.optimizer.params)
    pi = optimizer.solve(rel_mat_obs, expo)

    t0 = time()
    result = evaluate_pi(pi, rel_mat_true, expo)
    exec_time = time() - t0
    result.exec_time = exec_time

    print(result)

    wandb.log(
        {
            "user_util": result.user_util,
            "item_utils": result.item_utils,
            "max_envies": result.max_envies,
            "nsw": result.nsw,
            "exec_time": result.exec_time,
        }
    )


if __name__ == "__main__":
    main()
