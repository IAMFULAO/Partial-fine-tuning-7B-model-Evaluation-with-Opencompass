from mmengine.config import read_base

with read_base():
    from ...datasets.mmlu.mmlu_gen import mmlu_datasets as mmlu_gen_datasets
    from ...datasets.cmmlu.cmmlu_gen import cmmlu_alpaca_datasets
    from ...datasets.mmlu.mmlu_stem_0shot_cascade_eval_gen_216503 import mmlu_datasets as mmlu_stem_datasets
    from ...models.llama2.Llama2_7B_chat_lora import models

datasets = []
datasets.extend(mmlu_gen_datasets)
datasets.extend(cmmlu_alpaca_datasets)
datasets.extend(mmlu_stem_datasets)

work_dir = '/home/liruizheng/Partial_fine_tuning_task/opencompass/outputs/llama2_7b_chat_lora_eval'