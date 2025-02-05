import sys
sys.path.append('..')

import re
import pdb
import json
import argparse
from datasets import load_dataset
from jinja2 import Template
from easy_llm.models import ModelFactory


SYSTEM_PROMPT = '''你是一个医疗领域 NER 任务的标注专家。'''

USER_PROMPT_TEMPLATE = '''请识别出下面文档中「{{ entity_type }}」类别的实体：

「文档开始」
{{ text }}
「文档结束」

请严格按照列表格式输出，例如：[实体1,实体2,实体3]'''

ASSISTANT_PROMPT_TEMPLATE = '''{{ "[" + entities|join(", ") + "]" }}'''


def create_messages(sample, shots, use_cot):
    # 创建系统消息
    system_message = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    
    # 构建few-shot示例消息
    history_messages = []
    for shot in shots:
        entity_type = shot['entity_type']
        user_prompt = Template(USER_PROMPT_TEMPLATE).render(
            entity_type=entity_type, 
            text=shot['text']
        )
        user_message = [{'role': 'user', 'content': user_prompt}]
        assistant_prompt = Template(ASSISTANT_PROMPT_TEMPLATE).render(entities=json.loads(shot['entities']))
        assistant_message = [{'role': 'assistant', 'content': assistant_prompt}]
        history_messages.extend(user_message + assistant_message)
    
    # 构建当前样本的消息
    if 'instruction' in sample:
        user_prompt = sample['instruction']
    else:
        user_prompt = Template(USER_PROMPT_TEMPLATE).render(
            entity_type=sample['entity_type'], 
            text=sample['text']
        )
    if use_cot:
        user_prompt = '\n'.join(user_prompt.split('\n')[:-1]) + '\n请综合参考答案和自己的理解，先输出推理过程，再输出答案。最后输出的结果要严格按照列表格式，例如：[实体1,实体2,实体3]'
        # pdb.set_trace()
    user_message = [{'role': 'user', 'content': user_prompt}]
    
    return system_message + history_messages + user_message


# 解析实体列表（考虑CoT的输出格式）
def parse_entities(output):
    try:
        output = re.search(r'\[(.*?)\]$', output, re.DOTALL).group(1)
    except:
        return []
    if output == '':
        return []
    return [e.strip() for e in output.strip('[]').split(',')]


def make_prompt(messages, template_path):
    template = Template(open(template_path).read())
    return template.render(messages=messages)


# args: model_name, dataset_path, n_shots, temperature
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--n-shots', type=int, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--use-cot', type=str, choices=['true', 'false'], default='false', 
                       help='Whether to use Chain of Thought (CoT) strategy')
    parser.add_argument('--force-inference', type=str, choices=['true', 'false'], default='false', 
                       help='Whether to force inference')
    parser.add_argument('--tag', type=str, help='Tag for the inference')
    # parser.add_argument('--model-type', type=str, default='chat', choices=['chat', 'base'], help='Model type: chat for dialogue model, base for foundation model')
    parser.add_argument('--template-path', type=str, help='Path to prompt template file if using base model')
    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()
    data_files = [args.dataset_path]

    # 加载数据和模型
    dataset = load_dataset('json', data_files=data_files, split='train')
    factory = ModelFactory()
    model = factory.create_model(args.model_name)
    model.set_temperature(args.temperature)

    # dataset = predict(model, dataset, args.n_shots, args.use_cot == 'true', args.force_inference == 'true', args.tag)
    # 读取shots
    shots = load_dataset('json', data_files='shots.jsonl', split='train').to_list()
    # shots = []
    # 创建所有消息
    all_messages = [create_messages(sample, shots[:args.n_shots], args.use_cot == 'true') for sample in dataset]
    if model.model_type == 'chat':
        outputs = model.batch_inference(all_messages, force_inference=args.force_inference == 'true', tag=args.tag)
    else:
        all_prompts = [make_prompt(messages, args.template_path) for messages in all_messages]
        outputs = model.batch_inference(all_prompts, force_inference=args.force_inference == 'true', tag=args.tag)

    # 处理输出结果
    outputs = [parse_entities(output) for output in outputs]
    
    # 更新数据集
    if 'entities' in dataset.column_names:
        dataset = dataset.remove_columns(['entities'])
    dataset = dataset.add_column(name='entities', column=outputs)

    # 展示和保存
    # for sample in dataset.select(range(3)):
    #     print(f"{sample['text']}\n\n{sample['entity_type']}\n\n{sample['entities']}\n\n")
    safe_model_name = args.model_name.replace('/', '_')
    dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    save_path = f'data/{safe_model_name}_n{args.n_shots}_t{args.temperature}_cot-{args.use_cot}_tag-{args.tag}_{dataset_name}.jsonl'
    dataset.to_json(save_path, force_ascii=False)
    print(f'Saved to {save_path}')


if __name__ == "__main__":
    main()