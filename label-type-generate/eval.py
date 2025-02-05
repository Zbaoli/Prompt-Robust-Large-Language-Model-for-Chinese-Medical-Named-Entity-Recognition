import pdb
import re
from datasets import load_dataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path1', default='data/label_types_gpt3_5_turbo_v2.jsonl')
    parser.add_argument('--data-path2', default='data/label_types_gpt3_5_turbo_v3.jsonl')
    args = parser.parse_args()
    return args


def parse_labels(sample):
    text = sample['label_type']
    labels = re.findall(r'(\(.*?\))', text)
    labels = [eval(label) for label in labels]
    # labels = [label[0] for label in labels]
    return labels


def eval_consistency(dataset1, dataset2):
    correct, total = 0, 0
    for sample1, sample2 in zip(dataset1, dataset2):
        labels1 = parse_labels(sample1)
        labels2 = parse_labels(sample2)
        correct += len(set(labels1) & set(labels2))
        total += len(set(labels1) | set(labels2))
        print(f'correct: {set(labels1) & set(labels2)}')
        print(f'labels1: {set(labels1) - set(labels2)}')
        print(f'labels2: {set(labels2) - set(labels1)}')
        print()
        # print(f'total: {set(labels1) | set(labels2) - set(labels1) & set(labels2)}')
    print('consistency', correct / total)


def main():
    args = parse_args()
    gold_dataset = load_dataset('json', data_files=args.data_path1, split='train')
    pred_dataset = load_dataset('json', data_files=args.data_path2, split='train')
    eval_consistency(gold_dataset, pred_dataset)
    # eval_f1(gold_dataset, pred_dataset)


def eval_f1(gold_dataset, pred_dataset):
    correct = 0
    gold, pred = 0, 0
    for gold_sample, pred_sample in zip(gold_dataset, pred_dataset):
        gold_entities = gold_sample['entities']
        pred_entities = pred_sample['entities']
        correct += len(set(gold_entities) & set(pred_entities))
        gold += len(set(gold_entities))
        pred += len(set(pred_entities))
    recall = correct / gold
    precision = correct / pred
    f1 = 2 * precision * recall / (precision + recall)
    print('precision', precision)
    print('recall', recall)
    print('f1', f1)

if __name__ == '__main__':
    main()
