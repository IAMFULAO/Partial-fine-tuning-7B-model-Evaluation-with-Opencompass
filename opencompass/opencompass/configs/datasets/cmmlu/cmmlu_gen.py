from mmengine.config import read_base

with read_base():
    from .cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403
    from .cmmlu_gen_alpaca2 import cmmlu_datasets as cmmlu_alpaca_datasets  # noqa: F401, F403