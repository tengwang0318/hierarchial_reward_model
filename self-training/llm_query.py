from openai import OpenAI
from envs import *


def parallel_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0, n=NUMBER_OF_CHILDREN):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key, )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        stop=['# END!', '# Step 2', "# Step 3", "# Step 4", "# Step 5"],
        max_tokens=1024,
        extra_body={"include_stop_str_in_output": True, "repetition_penalty": repetition_penalty},
        n=n
    )
    return [choice.text for choice in completion.choices]


def sequential_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key, )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        stop=['# END!', '# Step 2', "# Step 3", "# Step 4", "# Step 5"],
        max_tokens=1024,
        extra_body={"include_stop_str_in_output": True, "repetition_penalty": repetition_penalty},

    )
    return completion.choices[0].text


def one_step_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key, )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        stop=['# END!'],
        max_tokens=3600,
        extra_body={"include_stop_str_in_output": True, "repetition_penalty": repetition_penalty},

    )
    return completion.choices[0].text
