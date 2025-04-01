from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams, inputs
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import torch.nn.functional as F


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


MODEL_PATH = "Qwen/Qwen2.5-Math-PRM-72B"
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

def make_input(question: str, stepwise_solution: str):
    return [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {'role': "assistant", "content": "<extra_0>".join(stepwise_solution) + "<extra_0>"}
    ]

chats = [make_input(q, s) for q, s in zip(dataset["problem"], dataset["stepwise_solution"])]
chats = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chats]

input_ids_list = [
    tokenizer.encode(
        chat, 
        return_tensors="pt", 
    ).to(model.device)
    for chat in chats
]

outputs_list = []

for input_ids in tqdm(input_ids_list):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)[0]
        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_reward = make_step_rewards(outputs, token_masks)
        outputs_list.append(step_reward)

dataset = dataset.add_column("prm_feedback", outputs_list)
dataset.push_to_hub("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math-prm", split="train")
