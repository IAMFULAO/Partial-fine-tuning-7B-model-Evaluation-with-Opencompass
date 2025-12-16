from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-2-7b-chat-lora',
        path='/home/liruizheng/models/Llama-2-7b-chat-lora-8*8loss=0.5',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_out_len=50,
        batch_size=8,
        batch_padding=False,
        run_cfg=dict(num_gpus=1),
    )
]