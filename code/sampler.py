from typing import Optional
import random
import requests
import json


class TGISampler(object):
    """
    Sample from TGI's completion API
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:8080",
        system_message: Optional[str] = None,
        temperature: float = 0.95,
        max_tokens: int = 2048,
        truncate: int = 7000,
    ):
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.truncate = truncate
        self.urls = url.strip().split('[SEP]')

    def get_resp(self, prompt, do_sample=False, temperature=None, topp=None):
        if do_sample:
            param = {
                "best_of": 1,
                "decoder_input_details": False,
                "details": False,
                "do_sample": True,
                "max_new_tokens": self.max_tokens,
                "return_full_text": False,
                "seed": 42,
                "temperature": temperature,
                "top_p": topp,
                # "truncate": self.truncate,
                "stop": ["<|user|>", "<|endoftext|>", "<|observation|>"],
            }
        else:
            param = {
                    "best_of": 1,
                    "decoder_input_details": False,
                    "details": False,
                    "do_sample": False,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False,
                    "seed": 42,
                    "stop": ["<|user|>", "<|endoftext|>", "<|observation|>"],
                }
        for i in range(5):
            try:
                url = random.choice(self.urls)
                rep = requests.post(
                    url,
                    json={
                        'inputs': prompt,
                        'stream': False,
                        "parameters": param
                    },
                    headers={
                        'Content-Type': 'application/json'
                    },
                    timeout=360
                )
                rep = rep.json()
                if isinstance(rep, list):
                    return rep[0]['generated_text'].strip().replace('<|user|>', '')
                else:
                    return rep['generated_text'].strip().replace('<|user|>', '')
            except Exception as e:
                print(f"Exception: ", e)
                continue
        return ''
    
    def dict_chat2prompt(self, message_list) -> str:
        prompt = ""
        for message in message_list:
            if message["role"] == "user":
                prompt += f"<|user|>\n{message['content']}"
            elif message["role"] == "assistant":
                prompt += f"<|assistant|>\n{message['content']}"
            elif message["role"] == "system":
                prompt += f"<|system|>\n{message['content']}"
        prompt += "<|assistant|>\n"
        return prompt

    def __call__(self, message_list, do_sample=False, temperature=0.5, topp=0.5) -> str:
        if self.system_message:
            message_list = [{"role": "system", "content": self.system_message}] + message_list
        prompt = self.dict_chat2prompt(message_list)
        return self.get_resp(prompt, do_sample=do_sample, temperature=temperature, topp=topp)



def query_chatglm_tgi(prompt, history=[], do_sample=True, max_tokens=2048, max_retry=3):
    # url = "http://172.18.64.8:8080/generate"
    url = "http://172.18.64.35:8080/generate"

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
            "decoder_input_details": True,
            "details": False,
            "do_sample": do_sample,
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            "seed": None,
            "temperature": 1,
            "top_p": 0.9,
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
