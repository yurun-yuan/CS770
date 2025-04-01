from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset, Dataset
import re


MODEL_PATH = "Qwen/Qwen2.5-Math-72B-Instruct"

dataset = load_dataset("meta-llama/Llama-3.1-8B-Instruct-evals", "Llama-3.1-8B-Instruct-evals__math__details", split="latest")

def map_fn(example):
    solution = example["output_prediction_text"][0]
    stepwise_solution = re.split(r'(?=## Step)', solution)
    if stepwise_solution and stepwise_solution[0].strip() == "":
        stepwise_solution = stepwise_solution[1:]

    return {
        "source": example["benchmark_label"],
        "problem": example["input_question"],
        "solution": solution,
        "stepwise_solution": stepwise_solution,
        "answer": example["output_parsed_answer"],
        "gt_answer": example["input_correct_responses"][0],
        "correctness": example["is_correct"]   
    }

dataset = dataset.map(map_fn, remove_columns=dataset.column_names)

dataset.push_to_hub("RyanYr/tutor-critic_llama-3.1-8b-instruct-evals-math", split="train")
