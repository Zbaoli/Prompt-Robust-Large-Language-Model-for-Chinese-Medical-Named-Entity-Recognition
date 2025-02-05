import sys
import re
sys.path.append('..')
from tqdm import tqdm
import pdb
import jinja2
import json
from collections import Counter
from datasets import load_dataset
from models import load_model
import itertools

gold_dataset = load_dataset("json", data_files="../annotate/annotated_data_multi.jsonl", split="train")
pred_dataset_names = [
    "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v1_chat.jsonl",
    # "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v2_chat.jsonl",
    "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v3_chat.jsonl",
    # "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v4_chat.jsonl",
    "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v5_chat.jsonl",
    # "data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v6_chat.jsonl",
]
pred_dataset_list = [
    load_dataset("json", data_files=dataset_name, split="train") for dataset_name in pred_dataset_names
]

RECHECK_TEMPLATE = '''请输出下列实体列表中属于「{{ entity_type }}」类别的实体：

{{ "[" + entities|join(", ") + "]" }}

请遵循上述格式输出，如遇到同名或同义实体，只需输出一个实体，如果没有实体，请输出空列表：'''

class RecheckAgent:
    def __init__(self):
        self.template = jinja2.Template(RECHECK_TEMPLATE)
        self.model = load_model('deepseek')
    
    def process(self, entity_type, entities):
        user_prompt = self.template.render(entity_type=entity_type, entities=entities)
        user_message = [{'role': 'user', 'content': user_prompt}]
        return self.model.inference(user_message)


def postprocess(sample, precision_first_entities, recall_first_entities):
    unknown_entities = set(recall_first_entities) - set(precision_first_entities)
    if len(unknown_entities) == 0:
        return []
    agent = RecheckAgent()
    reprocessed_entities = agent.process(sample['entity_type'], unknown_entities)

    def parse_entities(output):
        if output == '[]':
            return []
        output = re.search(r'\[(.*?)\]$', output).group(1)
        if output == '':
            pdb.set_trace()
        return [e.strip() for e in output.strip('[]').split(',')]

    return parse_entities(reprocessed_entities)


def remove_boundary_overlap(precision_first_entities, recall_first_entities):
    ans = []
    for entity1 in recall_first_entities:
        has_overlap = False
        for entity2 in precision_first_entities:
            if entity1 in entity2 or entity2 in entity1:
                has_overlap = True
                break
        if not has_overlap:
            ans.append(entity1)
    return ans


correct, gold, pred = 0, 0, 0
for i in tqdm(range(len(gold_dataset)), desc="Processing"):
    gold_entities = gold_dataset[i]["entities"]
    pred_entities_list = [pred_dataset[i]["entities"] for pred_dataset in pred_dataset_list]
    # precision-first
    num_models = len(pred_dataset_list)
    counter_entities = Counter(itertools.chain(*pred_entities_list))
    precision_first_entities = [entity for entity, count in counter_entities.items() if count == num_models]
    # recall-first
    recall_first_entities = set(itertools.chain(*pred_entities_list))
    print('entity type:', gold_dataset[i]['entity_type'])
    # for pred_entities in pred_entities_list:
    #     print('pred entities:', pred_entities)
    print('precision first entities:', precision_first_entities)
    print('recall first entities:', recall_first_entities)
    print('unknown entities:', recall_first_entities - set(precision_first_entities))
    print('boundary overlap:', remove_boundary_overlap(precision_first_entities, recall_first_entities))
    # postprocess
    checked_entities = postprocess(gold_dataset[i], precision_first_entities, recall_first_entities)
    print('checked entities:', checked_entities)
    print('')
    pred_entities = precision_first_entities + checked_entities
    correct += len(set(pred_entities) & set(gold_entities))
    gold += len(set(gold_entities))
    pred += len(set(pred_entities))


print(f"Precision = {correct / pred:.4f}, Recall = {correct / gold:.4f}, F1 = {2 * correct / (pred + gold):.4f}")
