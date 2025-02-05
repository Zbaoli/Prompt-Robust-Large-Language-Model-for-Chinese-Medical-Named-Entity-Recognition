from datasets import load_dataset

dataset = load_dataset('json', data_files='vanilla_train_4000.jsonl', split='train')
dataset = dataset.filter(lambda x: len(x['entities']) > 1)
print(len(dataset))

entity_types = set()
for sample in dataset:
    entity_types.add(sample['entity_type'])
print(entity_types)
print(len(entity_types))