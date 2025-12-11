from opencompass.datasets import C4Dataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLOnlyInferencer
from opencompass.openicl.icl_evaluator import AveragePPLEvaluator
c4_datasets = [
    dict(
        type=C4Dataset,  
        path='/home/liruizheng/datasets/llm_dataset/eval/c4',  
        abbr='c4',
        num=10,

        reader_cfg=dict(
            input_columns=['text'],   
            output_column=None,
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template='{text}',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=PPLOnlyInferencer,),
        ),
        
        eval_cfg=dict(evaluator=dict(type=AveragePPLEvaluator)),
    )
]