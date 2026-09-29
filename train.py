"""训练入口。

示例：
  python train.py --config configs/default.yaml
  python train.py --swanlab_mode local --debug_steps 2
"""

import sys

from src.arguments import parse_arguments
from src.trainer import OnPolicyTrainer
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    args = parse_arguments()
    logger.info("训练参数:\n%s", args)
    trainer = OnPolicyTrainer(args)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被手动中断")
        sys.exit(1)
