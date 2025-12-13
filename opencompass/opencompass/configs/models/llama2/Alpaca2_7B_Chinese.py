from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='alpaca-2-7b-Chinese',
        path='/home/liruizheng/models/Alpaca-2-7b-Chinese',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=50,
        batch_size=8,
        batch_padding=False,
        run_cfg=dict(num_gpus=1),
    )
]