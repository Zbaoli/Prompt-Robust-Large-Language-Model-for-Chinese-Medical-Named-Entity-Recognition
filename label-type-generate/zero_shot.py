import sys
import pdb
sys.path.append('..')

from tqdm import tqdm
from datasets import Dataset, load_dataset
from easy_llm.models import load_model
import random


SYSTEM_PROMPT_GUIDLINE = '''你是一个医疗领域 NER 任务的标注专家；'''

# PROMPT_CONTEXT = '''请抽取出下面文档的核心实体类别和定义，注意实体类别要是能明确定位的清晰明确的实体。
# 
# 「文档开始」
# {text}
# 「文档结束」
# 
# 请输出 json 格式，包含实体类别和定义。
# 如下所示：
# {{"label_type": "疾病", "definition": "请识别出文档中所有疾病名称相关的实体。"}}'''
# 
# UNINER_PROMPT = '''请抽取出下面文档中的实体以及他们的类别和类别定义。
# 
# 「文档开始」
# {text}
# 「文档结束」
# 
# 请按照下述格式输出：
# [("entity1", "type1 of entity1", "definition1 of type1"), ("entity2", "type2 of entity2", "definition2 of type2")]'''

UNINER_PROMPT_NO_DEFINITION = '''请抽取出下面文档中的实体以及他们的类别。

「文档开始」
{text}
「文档结束」

请按照下述格式输出：
[("entity1", "type1 of entity1"), ("entity2", "type2 of entity2")]'''


def generate_single_label_type(llm, sample, tag):
    # 根据 text 文档生成 label type
    system_message = {'role': 'system', 'content': SYSTEM_PROMPT_GUIDLINE}
    # 将当前样本的 text 作为 prompt 的输入
    # user_prompt = PROMPT_CONTEXT.format(text=sample['text'])
    user_prompt = UNINER_PROMPT_NO_DEFINITION.format(text=sample['text'])
    # user_prompt = UNINER_PROMPT.format(text=sample['text'])
    messages = [system_message, {'role': 'user', 'content': user_prompt}]
    outputs = llm.batch_inference([messages], tag=tag)
    return outputs[0]


def generate_label_types(llm, dataset, tag):
    batch_messages = []
    for sample in dataset:
        system_message = {'role': 'system', 'content': SYSTEM_PROMPT_GUIDLINE}
        user_prompt = UNINER_PROMPT_NO_DEFINITION.format(text=sample['text'])
        messages = [system_message, {'role': 'user', 'content': user_prompt}]
        batch_messages.append(messages)
    outputs = llm.batch_inference(batch_messages, tag=tag)
    # new_dataset = []
    # for sample, output in zip(dataset, outputs):
    #     sample['label_type'] = output
    #     new_dataset.append(sample)
    dataset = dataset.add_column('label_type', outputs)
    return dataset


def display_dataset(dataset):
    for sample in dataset:
        print(sample['text'])
        print(sample['label_type'])


def main():
    tag = 'v1'
    # data_file = '../data/mix/train.jsonl'
    data_file = '../data/mix/valid_1000.jsonl'
    dataset = load_dataset('json', data_files=data_file, split='train')
    dataset = dataset.filter(lambda x: len(x['text']) < 2000)
    # 0-100 作为测试集，1000-3000 作为训练集
    # dataset = dataset.shuffle(seed=42).select(range(1000, 2000))
    # model = load_model('deepseek')
    model = load_model('gpt-3.5-turbo-0125')
    dataset = generate_label_types(model, dataset, tag)
    dataset.to_json(f'data/label_types_gpt3_5_turbo_valid_1000_{tag}.jsonl', force_ascii=False)
    display_dataset(dataset)


if __name__ == "__main__":
    main()