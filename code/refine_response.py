
import random
import re
import os
from functools import partial
import pandas
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
from multiprocessing import Process, Queue
from tqdm import tqdm
import argparse
from sampler import TGISampler


SINGLE_PATTERN_TEMPLATE = """
In the following section, you will be presented with examples of math problems and their solutions. You will also receive a new math problem. Your task is to:

1. Based on the examples, summarize a potential reasoning pattern for the new math problem, including problem solving ideas and precautions, 
2. Refine the <Response> of the new problem based on the derived reasoning pattern.

### Examples of math problems and solutions:

{problem_prompt}

----------------------------

Please first find helpful patterns from the examples to solve the following new math problem without detailed calculations. Then rewrite the response based on the reasoning pattern in a step-by-step style.

### new math problem. 
<Question>
{question}

<Response>
{response}

Put the results in the following format:

[Summary]
<Lessons learned from examples that can help solve the new problem>

[Your analysis here]
<Your analysis here>

# Derived Reasoning Pattern
[Start of Reasoning Pattern]

[End of Reasoning Pattern]

# Refined Response
[Start of Refined Response]

[End of Refine Response]
"""



PATTERN_TEMPLATE = """
In the following section, you will be presented with examples of math problems and their solutions. You will also receive a new math problem. Your task is to:

Based on the examples, summarize a potential reasoning pattern for the new math problem, including problem solving ideas and precautions, 

### Examples of math problems and solutions:

{problem_prompt}

----------------------------

Find helpful patterns from the examples to solve the following new math problem.
Don't provide detailed solution and calculation. Just provide the reasoning idea and pattern.

### new math problem. 

{question}

Put the reasoning pattern in the following format:

[Summary]
<Lessons learned from examples that can help solve the new problem>

[Your analysis here]
<Your analysis here>

[Start of Reasoning Pattern]

[End of Reasoning Pattern]
"""


REFINE_TEMPLATE = """
A question and a reference answer will be provided below. Please respond like a student by writing out the problem-solving process in a step-by-step style, referring to the provided possible reasoning pattern and reference solution. Note that in your response, do not mention "reference idea / reference answer / reference response" etc.

### Possible Reasoning Pattern:
{pattern}

### Question: 
{prompt}

### Reference solution: 
{reference_response}

Problem-solving process:
"""


NAIVE_REFINE = """
A question and a reference answer will be provided below. Please respond like a student by writing out the problem-solving process  in a step-by-step style, referring to the provided solution. Note that in your response, do not mention "reference idea / reference answer / reference response" etc. 

### Question: 
{prompt}

### Reference solution: 
{reference_response}

### Problem-solving process:
"""


def extract_reasoning_pattern(output):
    if "[Start of Reasoning Pattern]" in output:
        return output.split("[Start of Reasoning Pattern]")[1].split("[End of Reasoning Pattern]")[0].strip()
    else:
        return output

def extract_response(output):
    if "[Start of Refined Response]" in output:
        return output.split("[Start of Refined Response]")[1].split("[End of Refine Response]")[0].strip()
    else:
        return output
    

def refine_with_pattern(sampler, pattern_sampler, problem_prompt, row: dict):
    prompt_key = "problem"
    response_key = "solution"
    
    prompt_messages = [dict(content=PATTERN_TEMPLATE.format(problem_prompt=problem_prompt, question=row[prompt_key]), role="user")]

    for i in range(0, 3):
        pattern_text = pattern_sampler(prompt_messages, do_sample=True, temperature=0.5, topp=0.5)

        try:
            pattern = extract_reasoning_pattern(pattern_text)
            # print("\n\n ----------- ")
            # print(pattern_text)

            break
        except Exception as e:
            print(e)
            pattern = "Let's think step by step"
            continue
    if len(pattern) > 16000 or pattern == "Let's think step by step":
        question_formatted = NAIVE_REFINE.format(prompt=row[prompt_key], reference_response=row[response_key])
    else:
        question_formatted = REFINE_TEMPLATE.format(pattern=pattern, prompt=row[prompt_key], reference_response=row[response_key])
    refined_response = sampler([dict(content=question_formatted, role="user")])
    row[response_key + "_refiend"] = refined_response
    return row


def naive_self_refine(sampler, row: dict):
    prompt_key = "problem"
    response_key = "solution"
    
    prompt_messages = [dict(content=NAIVE_REFINE.format(prompt=row[prompt_key], reference_response=row[response_key]), role="user")]

    refined_response = sampler(prompt_messages)
    row[response_key + "_refiend"] = refined_response
    return row


def process_worker(task_queue, done_queue, worker_func):
    max_retry = 3
    
    for line in iter(task_queue.get, "STOP"):
        result = worker_func(line)

        done_queue.put(result)

    done_queue.put("COMPLETE")


def map_with_progress(f: callable, xs: list[Any], num_threads: int = 50):
    num_processes = num_threads
    QUEUE_SIZE = 5000
    
    task_queue, done_queue = Queue(maxsize=QUEUE_SIZE), Queue(maxsize=QUEUE_SIZE)

    def read_data_into_queue():        
        for line in xs:
            task_queue.put(line)

        for _ in range(num_processes):
            task_queue.put('STOP')

    processes = []
    for _ in range(num_processes):
        process = Process(target=process_worker,
                    args=(task_queue, done_queue, f))
        process.start()
        processes.append(process)

    process = Process(target=read_data_into_queue)
    process.start()

    progress_bar = tqdm(total=len(xs))
    num_finished = 0
    
    results = []
    while num_finished < num_processes:
        item = done_queue.get()
        if item == 'COMPLETE':
            num_finished += 1
        else:
            if random.randint(0, 30) == 0:
                print(item)
            results.append(item)
            progress_bar.update(1)

    progress_bar.close()

    return results


def main_process(args):
    # load jsonl file
    data = []
    
    with open(args.input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # data = data[:100]
    print(f"Loaded {len(data)} data points")

    context_data = [json.loads(x) for x in open(args.context_data_path, "r")]
    random.seed(42)
    context_data = random.sample(context_data, args.num_shots)
    prompt_key = "instruction"
    response_key = "output"
    problem_prompt = "\n\n".join([f"Question: {x[prompt_key]}\n\nSolution\n{x[response_key]}" for x in context_data])
    
    sampler = TGISampler(args.sampler_url)
    pattern_sampler = TGISampler(args.pattern_sampler_url)
    
    if args.refine_mode == "pattern":
        results = map_with_progress(partial(refine_with_pattern, sampler, pattern_sampler, problem_prompt), data, args.num_process)
    else:
        results = map_with_progress(partial(naive_self_refine, sampler), data, num_threads=args.num_process)
    
    output_file = args.output_file.replace(".jsonl", f"_{args.refine_mode}.jsonl")
    with open(output_file, "w") as f:
        for line in results:
            f.write(json.dumps(line) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--sampler_url", type=str, required=True)
    parser.add_argument("--pattern_sampler_url", type=str, default=None)
    parser.add_argument("--refine_mode", type=str, default="pattern")
    parser.add_argument("--num_process", type=int, default=40)
    parser.add_argument("--context_data_path", type=str, default="../data/mathplus/mathplus-1k.jsonl")
    parser.add_argument("--num_shots", type=int, default=32)
    args = parser.parse_args()
    if args.pattern_sampler_url is None:
        args.pattern_sampler_url = args.sampler_url

    main_process(args)
