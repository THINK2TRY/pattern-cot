import json
import argparse
import requests
import random
# In the following you will be given some examples of math problemst / solutions and a set of principles about problem-solving. You will also be provided with a new math problem to solve. Your task is to read the problems and the solutions, select potential related principles to solve the new problem, and then summarize reasoning patterns to solve the new math problem based on the selected principles and providede problem-solution pairs. 


# TEMPLATE = """
# In the following section, you will be presented with examples of math problems and their solutions, along with a set of problem-solving principles. You will also receive a new math problem to solve. Your task is to:

# 1. Review the given problems and solutions.
# 2. Identify no more than 5 principles that can be applied to solve the new problem.
# 3. Summarize 1 or 2 reasoning patterns that can be helpful to solve the new math problem, based on the selected principles and provided problem-solution pairs.

# ### Examples of math problems and solutions:

# {problem_prompt}

# ----------------------------

# Please first understand and explain the key conditions and points of the following provided ### new math problem. 
# Then, summarize and design a reasoning pattern based on the given examples. 

# ### New math Problem:

# {question}

# """


TEMPLATE = """
In the following section, you will be presented with examples of math problems and their solutions. You will also receive a new math problem. Your task is to:

1. Review the given problems and solutions.
2. Identify no more than 3 principles that can be applied to solve the new problem.
3. Summarize 1 or 2 reasoning patterns that can be helpful to solve the new math problem, based on the selected principles and provided problem-solution pairs.

Put the reasoning pattern in the following format:
[Start of Reasoning Pattern]
    Proposed Reasoning Pattern
[End of Reasoning Pattern]

### Examples of math problems and solutions:

{problem_prompt}


### Potential useful principles

1. How can I break down this problem into smaller, more manageable parts?
2. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.
3. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.
4. Let’s think step by step.
5. Let’s make a step by step plan and implement it with good notion and explanation.
6. How can I simplify the problem so that it is easier to solve?
7. Carefully counting the numbers in your analysis when you have to know how many words/numbers are in the problem.


----------------------------

Please first understand and explain the key conditions and points of the following provided ### new math problem. 
Then, summarize and design a reasoning pattern based on the given examples. You don't need to provide the solution to the new problem.

{question}

"""



SOLUTION_TEMPLATE = """
Solve the following math problem step by step. You can consider using the provided reasoning pattern to solve the problem.
The last line of your response should be of the form ANSWER: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

### Problem
{question}

### Possible reasoning pattern:
{pattern}

"""



### Potential useful principles

# 1. How can I break down this problem into smaller, more manageable parts?
# 2. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.
# 3. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.
# 4. Let’s think step by step.
# 5. Let’s make a step by step plan and implement it with good notion and explanation.
# 6. How can I simplify the problem so that it is easier to solve?
# 7. Carefully counting the numbers in your analysis when you have to know how many words/numbers are in the problem.




# Finally, apply this pattern to solve the new math problem.

# 1. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.
# 2. How can I simplify the problem so that it is easier to solve?
# 3. What are the key assumptions underlying this problem?
# 4. What are the alternative perspectives or viewpoints on this problem?
# 5. How can I break down this problem into smaller, more manageable parts?
# 6. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.
# 7. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.
# 8. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the
# diverse perspectives and expertise of a group to come up with effective solutions.
# 9. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.
# 10. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.
# 11. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.
# 12. What is the core issue or problem that needs to be addressed?
# 13. What are the underlying causes or factors contributing to the problem?
# 14. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?
# 15. What are the potential obstacles or challenges that might arise in solving this problem?
# 16. How can progress or success in solving the problem be measured or evaluated?
# 17. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or
# theoretical problem?
# 18. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?
# 19. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing
# objectives?
# 20. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?
# 21. Does the problem require addressing systemic or structural issues rather than just individual instances?
# 22. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?
# 23. Ignoring the current best solution, create an entirely new solution to the problem.

# Please first repharase and understand the providede new probelm. Then summarize and design a reasoning pattern from the provided examples, then apply the pattern to solve the new math problem



def query_chatglm_platform(prompt, history=[], do_sample=True, max_tokens=128000):
    url = "http://172.18.64.92:9090/v1/chat/completions"

    messages = []
    for turn in history:
        messages.append({
            "role": "user",
            "content": turn["prompt"],
        })
        messages.append({
            "role": "assistant",
            "content": turn["response"],
        })
    messages.append({
        "role": "user",
        "content": prompt,
    })

    payload = {
        "messages": messages,
        "temperature": 0.9,
        "top_p": 0.7,
        # "model": self.model_version,
        "max_tokens": max_tokens,
        "do_sample": do_sample,
        "stream": False,
        "seed": random.randint(1, 10000000),
    }

    # response = requests.post(self.url, data=payload, headers=self.headers, verify=False)
    response = requests.post(url, json=payload, verify=False)
    
    if response.status_code == 200:
        answer = json.loads(response.text)
        # print(answer)
        # if answer["choices"][0]["finish_reason"] != "eos_token":
            # answer = None
        # else:
        answer = answer["choices"][0]["message"]["content"]
    else:
        print(response.text)
        answer = None

    return answer


def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3):
    url = "http://172.18.64.86:8080/generate" # ltx
    # url = "http://172.18.64.97:8080/generate"
    messages = ""
    for turn in history:
        ques, ans = turn["prompt"], turn["response"]
        messages += f"<|user|>\n{ques}<|assistant|>\n{ans}"

    messages += f"<|user|>\n{prompt}<|assistant|>\n"
    inputs = {
        "inputs": messages,
        "stream": False,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "seed": None,
            "temperature": 0.1,
            "top_p": 0.5,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        }
    }

    for _ in range(max_retry):
        output = requests.post(url, json=inputs)
        if output.status_code == 200:
            output = json.loads(output.text)
            # results.append(output[0]["generated_text"])
            result = output["generated_text"]
            break
        else:
            print(output.text)   
    else:
        result = None

    return result


def extract_reasoning_pattern(output):
    return output.split("[Start of Reasoning Pattern]")[1].split("[End of Reasoning Pattern]")[0].strip()
    

def main(args):
    context_data = [json.loads(x) for x in open(args.input_file)]
    prompt_key = "instruction"
    response_key = "output"
    problem_prompt = ""
    for i in range(args.num_shots):
        problem_prompt += f"Problem:\n{context_data[i][prompt_key]} \n\n Solution:\n{context_data[i][response_key]} \n\n"
    
    # question = context_data[499][prompt_key]
    # question = "Tom got a Mr. Potato Head for his birthday. It came with 3 hairstyles, 2 sets of eyebrows, 1 pair of googly eyes, 2 sets of ears, and 2 sets of lips, a pair of regular shoes, and a bonus pair of clown shoes. If a complete Mr. Potato Head personality includes eyebrows, eyes, ears, lips, shoes and optionally hair, how many different wacky personalities can Tom come up with? Note that Mr. Potato Head can be bald.\n\nNote: You cannot \"mix and match\".  For example, you cannot take the left eyebrow from one pair and the right eyebrow from the other pair."
    question = "The data in the stem and leaf plot shown are the long jump distances, in centimeters, that the girls team of Pseudo H.S. made at practice today. $(51|1$ represents $511$ centimeters$.)$ What is the sum of the median and mode of the data?\n\n\\begin{tabular}{l|lllll}\n51& 1\\\\\n52&\\\\\n53& 2& 5\\\\\n54& 0& 2& 2& 5\\\\\n55& 0& 1& 3& 4& 7\\\\\n56& 0& 2& 5\\\\\n57& 0& 1\\\\\n\\end{tabular}"
    
    query_template = TEMPLATE.format(problem_prompt=problem_prompt, question=question)
    print(f"Problem prompt: {question}")

    output = query_chatglm_tgi(query_template)

    print(f"\n\n### Output: \n\n")
    print(output)
    
    
def main_steps(args):
    context_data = [json.loads(x) for x in open(args.input_file)]
    prompt_key = "instruction"
    response_key = "output"
    problem_prompt = ""
    for i in range(args.num_shots):
        problem_prompt += f"Problem:\n{context_data[i][prompt_key]} \n\n Solution:\n{context_data[i][response_key]} \n\n"
    
    # question = context_data[499][prompt_key]
    # question = "Tom got a Mr. Potato Head for his birthday. It came with 3 hairstyles, 2 sets of eyebrows, 1 pair of googly eyes, 2 sets of ears, and 2 sets of lips, a pair of regular shoes, and a bonus pair of clown shoes. If a complete Mr. Potato Head personality includes eyebrows, eyes, ears, lips, shoes and optionally hair, how many different wacky personalities can Tom come up with? Note that Mr. Potato Head can be bald.\n\nNote: You cannot \"mix and match\".  For example, you cannot take the left eyebrow from one pair and the right eyebrow from the other pair."
    question = "The data in the stem and leaf plot shown are the long jump distances, in centimeters, that the girls team of Pseudo H.S. made at practice today. $(51|1$ represents $511$ centimeters$.)$ What is the sum of the median and mode of the data?\n\n\\begin{tabular}{l|lllll}\n51& 1\\\\\n52&\\\\\n53& 2& 5\\\\\n54& 0& 2& 2& 5\\\\\n55& 0& 1& 3& 4& 7\\\\\n56& 0& 2& 5\\\\\n57& 0& 1\\\\\n\\end{tabular}"
    
    query_template = TEMPLATE.format(problem_prompt=problem_prompt, question=question)
    print(f"Problem prompt: {question}")

    pattern = query_chatglm_tgi(query_template)
    pattern = extract_reasoning_pattern(pattern)
    pattern = "Think step by step.\n" + pattern
    print(f"\n\n### Reasoning Pattern: \n\n {pattern} \n\n")
    query = SOLUTION_TEMPLATE.format(pattern=pattern, question=question)
    output = query_chatglm_tgi(query)

    print(f"\n\n### Output: \n\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--num-shots", type=int, default=32)
    parser.add_argument("--input-file", type=str, default="../data/mathplus/mathplus-1k.jsonl")
    parser.add_argument("--output-file", type=str, default="../data/mathplus/output.json")
    args = parser.parse_args()
    main_steps(args)