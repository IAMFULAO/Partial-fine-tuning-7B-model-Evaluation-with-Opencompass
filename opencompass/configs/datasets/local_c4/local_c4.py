from opencompass.datasets import C4Dataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
c4_datasets = [
    dict(
        type=C4Dataset,  
        path='/home/liruizheng/datasets/llm_dataset/eval/c4',  
        name='validation', 
        abbr='c4',
        
        reader_cfg=dict(
            input_columns=['text'],  
            output_column='text',  
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template='{text}',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(
                type=PPLInferencer,
                splitting_method='tokenize',
                max_seq_len=512,
            ),
        ),
        
        eval_cfg=dict(evaluator=dict(type=AccEvaluator)),
    )
]
