import os
import json
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from lac import L2AdaptiveComputation
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


class EvalDataset(Dataset):

    def __init__(self, task):
        self.task = task

        if task == "MMLU":
            self.dataset = load_dataset("cais/mmlu", "all")

        elif task == "GPQA":
            self.dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]

        elif task == "BoolQ":
            self.dataset =  load_dataset("google/boolq")["validation"]

        else:
            raise ValueError(f"Unsupported task: {task}. Please choose from ['MMLU', 'GPQA', 'BoolQ'].")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.task == "MMLU":
            question = data["question"]
            choices = [data["choices"][i] for i in range(4)]
            correct_answer_idx = data["answer"]
            correct_letter = ["A", "B", "C", "D"][correct_answer_idx]

            prompt = format_prompt(question, choices)

            return prompt, correct_letter

        elif self.task == "GPQA":
            question = data["Question"]
            correct_answer = data["Correct Answer"]
            choices = [data[f"Incorrect Answer {i + 1}"] for i in range(3)]

            prompt = format_prompt(question, choices)

            return prompt, correct_answer

        else:
            question = data["question"]
            correct_answer = data["answer"]
            passage = data["passage"]

            prompt = format_boolq_prompt(question, passage)

            return prompt, correct_answer

def format_prompt(question, choices):
    options = ["A", "B", "C", "D"]
    formatted_choices = "\n".join([f"{options[i]}. {choice}" for i, choice in enumerate(choices)])

    prompt = [
        {"role": "user", "content": f"""
    Question:
    {question}

    Options:
    {formatted_choices}

    Output format:
    [Option]

    """}]

    return prompt


def format_boolq_prompt(question, passage):
    prompt = [{"role": "user", "content": f"""
    Respond to the given question based on the passage with True or False.
    Passage:
    {passage}

    Options:
    {question}

    Output Format:
    [True/False]
    """}]
    return prompt

def evaluate(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"


    eval_dataset = EvalDataset(args.task)

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    print(next(iter(eval_dataloader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on GPQA, MMLU, or BoolQ with L2 Adaptive Computation.")


    parser.add_argument("--dataset", type=str, required=True, choices=['GPQA', 'MMLU', 'BoolQ'],
                        help="Task to evaluate LMs on (GPQA, MMLU, or BoolQ).")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing examples.")

    args = parser.parse_args()

    evaluate(args)