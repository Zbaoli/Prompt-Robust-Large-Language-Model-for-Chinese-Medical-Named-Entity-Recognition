import sys
sys.path.append('..')

import re
import pdb
import jinja2
from tqdm import tqdm
import json
from collections import Counter
from datasets import load_dataset
from models import load_model


SYSTEM_PROMPT = '''你是一个NER领域的实体标注专家，请识别并输出文档中的指定实体；
你的输出必须严格按照列表格式：
「输出格式参考」
[感冒,发烧,头疼]'''

VANILLA_TEMPLATE = '''请识别文档中「{{ entity_type }}」类别的实体：

「文档开始」
{{ text }}
「文档结束」

「参考答案开始」
{% for answer in reference_answers %}
参考答案{{ loop.index }}: {{ "[" + answer|join(", ") + "]" }}
{% endfor %}
「参考答案结束」

上述参考答案来自不同模型，仅供参考。
请综合参考答案和自己对实体定义的理解，按照相同格式输出上述文档中符合实体定义的实体。'''

COT_TEMPLATE = '''请识别文档中「{{ entity_type }}」类别的实体：

「文档开始」
{{ text }}
「文档结束」

「参考答案开始」
{% for answer in reference_answers %}
参考答案{{ loop.index }}: {{ "[" + answer|join(", ") + "]" }}
{% endfor %}
「参考答案结束」

上述参考答案来自不同模型，仅供参考。
请综合参考答案和自己的理解，先输出推理过程，再按照上述格式输出答案。'''

SAME_ENTITY_PROMPT = '''请判断下面的实体哪些实体名字不同，但是是相同的实体：
{{ entities }}
输出格式是：
感冒：[]'''

class VanillaAgent:
    def __init__(self):
        self.template = jinja2.Template(VANILLA_TEMPLATE)
        self.model = load_model('deepseek')
    
    def generate_labels(self, text, label_type, reference_answers: dict):
        user_prompt = self.template.render(text=text, label_type=label_type, reference_answers=reference_answers)
        user_message = [{'role': 'user', 'content': user_prompt}]
        system_message = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages = system_message + user_message
        return self.model.inference(messages, temperature=0.2)


class ChainOfThoughtAgent:
    def __init__(self):
        self.template = jinja2.Template(COT_TEMPLATE)
        self.model = load_model('deepseek')

    def generate_labels(self, text, label_type, reference_answers: dict):
        user_prompt = self.template.render(text=text, label_type=label_type, reference_answers=reference_answers)
        user_message = [{'role': 'user', 'content': user_prompt}]
        system_message = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages = system_message + user_message
        return self.model.inference(messages, temperature=0.2)


class VotingAgent:
    def generate_labels(self, text, label_type, reference_answers: list):
        entities = []
        for answer in reference_answers:
            entities.extend(answer)
        num_models = len(reference_answers)
        counter = Counter(entities)
        voting_entities = [entity for entity, count in counter.items() if count >= num_models / 2]
        return '[' + ', '.join(voting_entities) + ']'


class SameEntityDetectionAgent:
    def __init__(self):
        self.model = load_model('deepseek')

    def generate_labels(self, text, label_type, reference_answers: dict):
        pass


def main():
    # agent = ChainOfThoughtAgent()
    # agent = VanillaAgent()
    agent = VotingAgent()
    gold_dataset = load_dataset('json', data_files='test.jsonl', split='train')
    pred_dataset_names = [
        "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v1_chat.jsonl",
        "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v3_chat.jsonl",
        "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v4_chat.jsonl",
        "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v5_chat.jsonl",
        "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v6_chat.jsonl",
    ]
    pred_dataset_list = [load_dataset('json', data_files=dataset_name, split='train') for dataset_name in pred_dataset_names]

    n = len(gold_dataset)
    # n = 3
    outputs = []
    for i in tqdm(range(n)):
        gold_entities = gold_dataset[i]["entities"]
        pred_entities_list = [pred_dataset[i]["entities"] for pred_dataset in pred_dataset_list]

        reference_answers = pred_entities_list
        labels = agent.generate_labels(gold_dataset[i]['text'], gold_dataset[i]['entity_type'], reference_answers)
        outputs.append(labels)

    def parse_entities(output):
        output = re.search(r'\[(.*?)\]$', output).group(1)
        if output == '':
            pdb.set_trace()
        return [e.strip() for e in output.strip('[]').split(',')]

    outputs = [parse_entities(output) for output in outputs]
    outputs = [json.dumps(output, ensure_ascii=False) for output in outputs]

    gold_dataset = gold_dataset.select(range(n))
    gold_dataset = gold_dataset.remove_columns(['entities'])
    gold_dataset = gold_dataset.add_column(name='entities', column=outputs)
    gold_dataset.to_json('data/direct_voting.jsonl', force_ascii=False)


if __name__ == '__main__':
    main()