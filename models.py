import sqlite3
import json
import pdb
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from abc import ABC, abstractmethod
from openai import OpenAI
import jinja2
from concurrent.futures import ThreadPoolExecutor


CACHE_PATH = ''

local_model_metadata = {
    'qwen2.5-7b-chat': {
        'path': '',
        'num_gpu': 1,
        'model_type': 'chat',
    }
}

openai_compatible_metadata = {
    'deepseek': {
        'api_key': '',
        'base_url': 'https://api.deepseek.com',
        'model_name': 'deepseek-chat'
    },
}


class MetaModel(ABC):
    @abstractmethod
    def batch_inference(self, batch_messages):
        pass


class VllmChatModel(MetaModel):
    def __init__(self, model_path, num_gpu, model_type):
        self.llm = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=num_gpu, max_num_seqs=100, gpu_memory_utilization=0.95, max_model_len=4096, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(max_tokens=1024, temperature=1.0, top_k=50, top_p=1.0)
        self.model_type = model_type

    def generate(self, prompts, temperature=0.0):
        self.sampling_params.temperature = temperature
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def chat(self, messages, temperature=0.0):
        self.sampling_params.temperature = temperature
        outputs = self.llm.chat(messages, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

    def batch_inference(self, inputs, temperature=0.0, force_inference=True, tag='v1'):
        # 支持纯文本输入和对话输入，纯文本输入使用 generate 方法，对话输入使用 chat 方法
        if isinstance(inputs[0], str):
            print('get pure text inputs, use generate method')
            return self.generate(inputs, temperature=temperature)
        else:
            print('get chat inputs, use chat method')
            return self.chat(inputs, temperature=temperature)
    
    def inference(self, input):
        return self.batch_inference([input])[0]

    def set_temperature(self, temperature):
        self.sampling_params.temperature = temperature


class ResultCache:
    """
    用于缓存模型推理结果的类。

    该类使用SQLite数据库来存储和管理模型的推理结果,以避免重复计算。
    每个结果都与特定的输入消息、温度参数和标签相关联。

    属性:
        db: SQLite数据库连接对象
        cursor: 数据库游标对象

    主要功能:
        - 根据输入消息、温度和标签存储/获取推理结果
        - 支持批量查询结果是否存在
        - 提供事务提交控制
    """
    def __init__(self, model_name):
        safe_model_name = model_name.replace('/', '_')
        self.db = sqlite3.connect(os.path.join(CACHE_PATH, f'{safe_model_name}_results.db'))
        self.cursor = self.db.cursor()
        self._init_table()
    
    def _init_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER, 
                messages TEXT,
                result TEXT,
                temperature REAL, 
                tag TEXT,
                PRIMARY KEY (id, temperature, tag)
            )
        ''')
        self.db.commit()

    def get_result(self, messages, temperature, tag):
        self.cursor.execute(
            '''SELECT result FROM results WHERE messages = ? AND temperature = ? AND tag = ?''', 
            (json.dumps(messages, ensure_ascii=False), temperature, tag)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def exists(self, messages, temperature, tag):
        self.cursor.execute(
            '''SELECT COUNT(*) FROM results WHERE messages = ? AND temperature = ? AND tag = ?''', 
            (json.dumps(messages, ensure_ascii=False), temperature, tag)
        )
        return self.cursor.fetchone()[0] > 0

    def save_result(self, messages, result, temperature, tag, do_commit=True):
        self.cursor.execute(
            '''INSERT INTO results (messages, result, temperature, tag) VALUES (?, ?, ?, ?)''', 
            (json.dumps(messages, ensure_ascii=False), result, temperature, tag)
        )
        if do_commit:
            self.db.commit()

    def commit(self):
        self.db.commit()

    def bulk_exists(self, messages_json, temperature, tag):
        # 批量查询 messages 是否存在
        query = "SELECT messages FROM results WHERE messages IN ({}) AND temperature = ? AND tag = ?".format(','.join('?' for _ in messages_json))
        self.cursor.execute(query, messages_json + [temperature, tag])
        existing_messages = {row[0] for row in self.cursor.fetchall()}
        return [message in existing_messages for message in messages_json]

    def save_results_bulk(self, batch_data, do_commit=True):
        # 批量插入数据
        self.cursor.executemany(
            '''INSERT INTO results (messages, result, temperature, tag) VALUES (?, ?, ?, ?)''', 
            batch_data
        )
        if do_commit:
            self.db.commit()


class OpenAIModel(MetaModel):
    def __init__(self, api_key, base_url, model_name, temperature=0.0, batch_size=30) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.cache = ResultCache(model_name)
        self.temperature = temperature
        self.batch_size = batch_size

    def batch_inference(self, batch_messages, force_inference=False, tag='v1'):
        outputs = []
        # 批量检查缓存
        messages_json = [json.dumps(messages, ensure_ascii=False) for messages in batch_messages]
        exists = self.cache.bulk_exists(messages_json, self.temperature, tag)
        # concurrent inference
        outputs = self._concurrent_inference(batch_messages, self.temperature, force_inference, tag, exists)
        # prepare data for bulk insert
        batch_data = []
        for i, (messages, result, exist) in tqdm(enumerate(zip(batch_messages, outputs, exists)), desc="Preparing data for db..."):
            if not exist:
                batch_data.append((json.dumps(messages, ensure_ascii=False), result, self.temperature, tag))
            else:
                outputs[i] = self.cache.get_result(messages, self.temperature, tag)
        # save all results in bulk
        self.cache.save_results_bulk(batch_data)
        return outputs

    def _concurrent_inference(self, batch_messages, force_inference, tag, exists):
        outputs = []
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = [executor.submit(self._inference, messages, self.temperature, force_inference, tag, exist) for messages, exist in zip(batch_messages, exists)]
            for future in tqdm(futures, desc='Inferring', total=len(batch_messages)):
                outputs.append(future.result())
        return outputs

    def _inference(self, messages, force_inference, tag, exist):
        # if messages in db, 则直接从 db 中读取
        # 如果 force_inference 为 True，则不从 db 中读取
        if not force_inference and exist:
            return ''
        else:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return completion.choices[0].message.content

    def inference(self, messages):
        return self.batch_inference([messages], force_inference=True)[0]

    def chat(self, messages):
        return self.inference(messages)

    def set_temperature(self, temperature):
        self.temperature = temperature


class ModelFactory:
    def __init__(self):
        self.local_model_metadata = local_model_metadata
        self.openai_compatible_metadata = openai_compatible_metadata

    def create_model(self, model_name):
        # Check if model exists in either metadata
        if model_name not in self.local_model_metadata and model_name not in self.openai_compatible_metadata:
            raise ValueError(f"Model {model_name} not found in any metadata.")
        
        # Check local models first
        if model_name in self.local_model_metadata:
            metadata = self.local_model_metadata[model_name]
            return VllmChatModel(
                model_path=metadata['path'], 
                num_gpu=metadata['num_gpu'], 
                model_type=metadata['model_type']
            )
        
        # Then check OpenAI compatible models
        metadata = self.openai_compatible_metadata[model_name]
        return OpenAIModel(
            api_key=metadata['api_key'],
            base_url=metadata['base_url'], 
            model_name=metadata['model_name']
        )


if __name__ == '__main__':
    # Create model factory
    factory = ModelFactory()
    
    # Test creating and using local model
    local_model = factory.create_model('qwen2.5-7b-chat')
    
    # Test single message inference
    message = [{'role': 'user', 'content': 'What is 2+2?'}]
    print("Single message test:")
    print(local_model.inference(message))
    print()
    
    # Test batch message inference
    batch_messages = [
        [{'role': 'user', 'content': f'Count to {i}'}] 
        for i in range(1,4)
    ]
    print("Batch messages test:")
    outputs = local_model.batch_inference(batch_messages)
    for i, output in enumerate(outputs, 1):
        print(f"Query {i}:", output)
        print()
    
    # Test temperature setting
    print("Testing temperature change:")
    local_model.set_temperature(0.8)
    print(local_model.inference(message))
