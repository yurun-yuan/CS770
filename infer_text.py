from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams, inputs
import torch
from datasets import load_dataset, Dataset, concatenate_datasets


MODEL_PATH = "Qwen/Qwen2.5-Math-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset = load_dataset("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math", split="train")
def sample(dataset, correct_n, incorrect_n):
    correct = dataset.filter(lambda x: x["correctness"])
    incorrect = dataset.filter(lambda x: not x["correctness"])
    correct = correct.shuffle(seed=42).select(range(correct_n))
    incorrect = incorrect.shuffle(seed=42).select(range(incorrect_n))
    return concatenate_datasets([correct, incorrect])

dataset = sample(dataset, 200, 200)

def make_input(question: str, solution: str):
    return [
    {'role': 'system', 'content': (
        'You are an expert grader. Review the student’s solution carefully. '
        'Clearly identify any inaccuracies or logical errors, but do NOT provide the correct solution.'
    )},
    {'role': 'user', 'content': f'''Problem:
{question}

Student's solution:
{solution}

Clearly identify any inaccuracies or logical errors, but do NOT provide the correct solution.
'''}
]

chats = [make_input(q, s) for q, s in zip(dataset["problem"], dataset["solution"])]
chats = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True) for chat in chats]

MAX_LEN = 4096
chats = [input for input in chats if len(tokenizer.encode(input)) <= MAX_LEN]

llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=4096,
    load_format="auto",
    seed=42,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=torch.cuda.device_count(),
    swap_space=0, # Only if n in sampling_params is 1
)

sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=2048,
    n=1,
    skip_special_tokens=False,
    ignore_eos=False,
    seed=42,
)

raw_outputs = llm.generate(chats,  sampling_params=sampling_params, use_tqdm=True)
outputs = [raw_output.outputs[0].text for raw_output in raw_outputs]

dataset = dataset.add_column("text_feedback", outputs)
dataset.push_to_hub("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math-text_feedback")
