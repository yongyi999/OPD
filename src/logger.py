"""SwanLab 实验记录封装。

- mode=online：实验同步到 SwanLab 云端（需先 `swanlab login` 或设置
  环境变量 SWANLAB_API_KEY）
- mode=local：只写本地，可用 `swanlab watch <logdir>` 在浏览器查看
- mode=offline：仅本地记录，不可视化
- mode=disabled：完全关闭
"""

from .utils import get_logger

logger = get_logger(__name__)


class SwanLabLogger:
    """对 swanlab API 的薄封装，任何异常都不应中断训练。"""

    def __init__(self, args, config_dict: dict):
        self.mode = args.swanlab_mode
        self.enabled = self.mode != "disabled"
        self.run = None

        if not self.enabled:
            logger.info("SwanLab 已关闭（mode=disabled）")
            return

        try:
            import swanlab

            kwargs = dict(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment_name or None,
                config=config_dict,
                mode=self.mode,
                logdir=args.swanlab_logdir,
            )
            if args.swanlab_entity:
                kwargs["entity"] = args.swanlab_entity

            self.run = swanlab.init(**kwargs)
            logger.info(
                "SwanLab 已初始化（mode=%s, project=%s）",
                self.mode,
                args.swanlab_project,
            )
        except Exception as e:  # 记录失败不影响训练
            logger.warning("SwanLab 初始化失败，后续只打印到控制台: %s", e)
            self.enabled = False

    def log(self, metrics: dict, step: int = None):
        """记录标量指标。"""
        if not self.enabled:
            return
        try:
            import swanlab

            if step is not None:
                swanlab.log(metrics, step=step)
            else:
                swanlab.log(metrics)
        except Exception as e:
            logger.warning("SwanLab 记录失败: %s", e)

    def log_text(self, tag: str, text: str, step: int = None):
        """记录文本（如学生生成样例）。"""
        if not self.enabled:
            return
        try:
            import swanlab

            payload = {tag: swanlab.Text(text)}
            if step is not None:
                swanlab.log(payload, step=step)
            else:
                swanlab.log(payload)
        except Exception as e:
            logger.warning("SwanLab 文本记录失败: %s", e)

    def finish(self):
        """结束实验。"""
        if not self.enabled:
            return
        try:
            import swanlab

            swanlab.finish()
        except Exception as e:
            logger.warning("SwanLab finish 失败: %s", e)
