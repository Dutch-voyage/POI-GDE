import logging
import os
import sys
import traceback

import hydra
from omegaconf import OmegaConf
from trainers.ODETrainer import ODETrainer
from trainers.TestTrainer import TestTrainer

@hydra.main(config_path='configs', config_name='ode.yaml', version_base='1.2')
def main(config: OmegaConf):
    try:
        logging.info(OmegaConf.to_yaml(config))
        # OmegaConf.save(config, os.path.join(os.getcwd(), 'config.yaml'))
        checkpoints_dir = os.path.join(os.getcwd(), 'ckpts')
        os.makedirs(checkpoints_dir, exist_ok=True)
        kwargs = OmegaConf.to_container(config, resolve=True)
        trainer = ODETrainer(**kwargs)
        # trainesr = TestTrainer(**kwargs)
        trainer.fit()
    except KeyboardInterrupt:
        logging.warning('Interrupted by user')
    except Exception as ex:
        logging.critical(f'Training failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

if __name__ == '__main__':
    main()