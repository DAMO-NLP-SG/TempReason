import os
from typing import List
import json
import torch
#os.environ['WANDB_MODE'] = 'offline'


os.environ["WANDB_DISABLED"] = "true"


#os.environ['WANDB_MODE'] = 'disabled'
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_metric
import trlx
from trlx.data.configs import TRLConfig
import datasets

try:
    import evaluate
except ImportError:
    raise ImportError(
        "To run this example, please install the `evaluate` and `nltk` packages"
        "by running `pip install evaluate`"
    )

config_path = os.path.join(
    os.path.dirname(__file__), "configs/ppo_config_qa.yml"
)
config = TRLConfig.load_yaml(config_path)
metric =  load_metric("/mnt/workspace/project/datasets/metrics/squad/squad.py")
metric.features['references']['answers'] = datasets.features.Sequence(feature={'text': datasets.Value(dtype='string', id=None)})

if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        prompts = prompts
        predictions = [
            {
                'id':str(i),
                'prediction_text': output,
            } for i, output in enumerate(outputs)
        ]
        references = [
            {
            'id':str(i),
            'answers': {'text': prompt_label[prompt.strip()]},
            } for i, prompt in enumerate(prompts)
        ]
        negative_references = [
            {
            'id':str(i),
            'answers': {'text': prompt_neg_label[prompt.strip()]},
            } for i, prompt in enumerate(prompts)
        ]


        rewards = []
        for (pred, gt, neg_ans) in zip(predictions, references, negative_references):
            ref_score = metric.compute(predictions=[pred], references=[gt])['exact_match']/100
            neg_score = metric.compute(predictions=[pred], references=[neg_ans])['exact_match']/100
            if (ref_score==0) and (neg_score==0):
                rewards.append(-2.0)
            elif (ref_score >= neg_score) and (ref_score!=0):
                rewards.append(1.0 * ref_score)
            elif ref_score < neg_score:
                rewards.append(-1.0 * neg_score)

        return rewards
    def metric_fn(samples: List[str], prompts: List[str], outputs: List[str]) -> List[float]:
        """Compute EM and F1"""

        prompts = prompts
        predictions = [
            {
                'id':str(i),
                'prediction_text': output,
            } for i, output in enumerate(outputs)
        ]
        references = [
            {
            'id':str(i),
            'answers': {'text': prompt_label[prompt.strip()]},
            } for i, prompt in enumerate(prompts)
        ]


        scores = metric.compute(predictions=predictions, references=references)
        f1_score = scores['f1']
        em_score = scores['exact_match']
        return {"em": em_score, "f1": f1_score}
    from datasets import load_dataset
    dlc_path = '/mnt/workspace/project/custom_transformers/examples/pytorch/question-answering/temp_reason/'    
    train_data = [json.loads(line) for line in open(dlc_path + 'train_l2.json')]
    val_data = [json.loads(line) for line in open(dlc_path + 'test_l2.json')]
    val_size = len(val_data)
    data = train_data + val_data
    prompts = []
    samples = ()
    rewards = ()
    answers = []
    neg_answers = []

    for _, example in enumerate(data):
        question = example['question']
        context = example['fact_context']
        prompt = ' '.join([question, context])
        #print(example['text_answers']['text'])
        answer = example['text_answers']['text']
        neg_answer = example['neg_answers']
        answers.append(answer)
        prompts.append(prompt)
        neg_answers.append(neg_answer)
    print(config.model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, padding_side="right", truncation_side="right",)
    print(len(tokenizer))
    train_prompts = prompts[:-val_size]
    val_prompts = prompts[-val_size:]
    prompt_label = {}
    prompt_neg_label = {}
    max_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]
    prompt_tokens = []
    for i in range(len(prompts)):
        # if prompts[i].startswith("London (CNN)At the time it probably seemed like fun: Jeremy Clarkson and"):
        #     import ipdb; ipdb.set_trace()
        key = tokenizer.decode(
            tokenizer(
                prompts[i],
                truncation=True,
                max_length=max_length
            )['input_ids'],
            skip_special_tokens=True, 
        )
        value = [tokenizer.decode(
            tokenizer(
                x,
                truncation=True,
                max_length=max_length
            )['input_ids'],
            skip_special_tokens=True, 
        ) for x in answers[i]]
        neg_value = [tokenizer.decode(
            tokenizer(
                x,
                truncation=True,
                max_length=max_length
            )['input_ids'],
            skip_special_tokens=True, 
        ) for x in neg_answers[i]]
        prompt_label[key.strip()] = value
        prompt_neg_label[key.strip()] = neg_value

            



    results, predictions = trlx.evaluate(
        config.model.model_path,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        prompts=train_prompts,
        eval_prompts=val_prompts,
        config=config,
    )

    json.dump(predictions, open('predictions.json', 'w'))

