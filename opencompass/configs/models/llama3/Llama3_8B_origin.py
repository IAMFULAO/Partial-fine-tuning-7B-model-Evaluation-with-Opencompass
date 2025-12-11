from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-3-8b',
        path='/home/liruizheng/models/Llama-3-8B/._____temp/LLM-Research/Meta-Llama-3-8B',
        tokenizer_path='/home/liruizheng/models/Llama-3-8B/._____temp/LLM-Research/Meta-Llama-3-8B',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        batch_size=8,
        batch_padding=False,
        run_cfg=dict(num_gpus=1),
    )
]