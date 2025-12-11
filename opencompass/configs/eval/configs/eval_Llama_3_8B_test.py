from mmengine.config import read_base

with read_base():
    from ...datasets.local_c4.local_c4_test import c4_datasets  
    from ...models.llama3.Llama3_8B_origin import models 

datasets = [*c4_datasets]
work_dir = '/home/liruizheng/opencompass/outputs/llama3_8b_eval'