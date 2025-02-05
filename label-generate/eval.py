import pdb
from datasets import load_dataset
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path1', default='annotated_data_multi.jsonl')
    parser.add_argument('--data-path2', default='data/gpt-3.5-turbo-0125_n0_t0.0_cot-false_tag-v4_chat.jsonl')
    # parser.add_argument('--data-path2', default='data/label_types_gpt3_5_turbo_annotated.jsonl')
    # parser.add_argument('--data-path2', default='data/vanilla_n0_t0.0_chat.jsonl')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gold_dataset = load_dataset('json', data_files=args.data_path1, split='train')
    pred_dataset = load_dataset('json', data_files=args.data_path2, split='train')
    eval_f1(gold_dataset, pred_dataset)
    eval_correct_format(pred_dataset)


def load_sample_entities(sample):
    if 'entities' in sample:
        entities = sample['entities']
    else:
        entities = sample['output']
    if isinstance(entities, str):
        entities = parse_entities(entities)
    return entities

def parse_entities(output):
    # "[白萝卜汤, 中药]"
    try:
        # 尝试匹配最后一个方括号中的内容
        output = output.strip()
        if output == '[]':
            return []
        # 使用正则表达式匹配最后一个方括号中的内容
        import re
        match = re.search(r'\[(.*?)\]$', output, re.DOTALL)
        if not match:
            return []
        output = match.group(1)
        # 分割并清理实体
        entities = [e.strip() for e in output.split(',')]
        # 过滤掉空字符串
        entities = [e for e in entities if e]
        return entities
    except:
        return []


def eval_f1(gold_dataset, pred_dataset):
    correct = 0
    gold, pred = 0, 0
    for gold_sample, pred_sample in zip(gold_dataset, pred_dataset):
        gold_entities = load_sample_entities(gold_sample)
        pred_entities = load_sample_entities(pred_sample)
        correct += len(set(gold_entities) & set(pred_entities))
        gold += len(set(gold_entities))
        pred += len(set(pred_entities))
    recall = correct / gold
    precision = correct / pred
    f1 = 2 * precision * recall / (precision + recall)
    print('precision', precision)
    print('recall', recall)
    print('f1', f1)

def eval_correct_format(dataset):
    correct = 0
    for sample in dataset:
        if len(sample['entities']) > 0:
            correct += 1
    print('correct format accuracy', correct / len(dataset))


if __name__ == '__main__':
    main()
