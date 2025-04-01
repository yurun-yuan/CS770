from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams, inputs
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



MODEL_PATH = "Qwen/Qwen2.5-Math-RM-72B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset = load_dataset("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math", split="train")
def sample(dataset, correct_n, incorrect_n):
    correct = dataset.filter(lambda x: x["correctness"])
    incorrect = dataset.filter(lambda x: not x["correctness"])
    correct = correct.shuffle(seed=42).select(range(correct_n))
    incorrect = incorrect.shuffle(seed=42).select(range(incorrect_n))
    return concatenate_datasets([correct, incorrect])

dataset = sample(dataset, 200, 200)

model = AutoModel.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

def make_input(question: str, solution: str):
    return [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {'role': "assistant", "content": solution}
    ]

chats = [make_input(q, s) for q, s in zip(dataset["problem"], dataset["solution"])]
chats = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

input_ids_list = [
    tokenizer.encode(
        chat, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    for chat in chats
]

outputs = []

for input_ids in tqdm(input_ids_list):
    with torch.no_grad():
        output = model(input_ids=input_ids)[0].squeeze().item()
        output = sigmoid(output)
        outputs.append(output)

dataset = dataset.add_column("rm_feedback", outputs)
dataset.push_to_hub("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math-rm", split="train")
