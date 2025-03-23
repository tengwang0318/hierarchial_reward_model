from openai import OpenAI



def orm_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key)

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        max_tokens=2048,
        stop=['# END!'],

        extra_body={"repetition_penalty": repetition_penalty, "include_stop_str_in_output": False}
    )

    return completion.choices[0].text


def orm_parallel_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0, n=1):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key)

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        max_tokens=2048,
        stop=['# END!'],

        extra_body={"repetition_penalty": repetition_penalty, "include_stop_str_in_output": False},
        n=n

    )

    return [choice.text for choice in completion.choices]


def prm_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key, )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        stop=['\]\n\n', '\)\n\n', '# END!'],
        max_tokens=1024,
        extra_body={"include_stop_str_in_output": True, "repetition_penalty": repetition_penalty}

    )

    return completion.choices[0].text


def prm_parallel_query(host, port, model_name, prompt, api_key="", repetition_penalty=1.0, n=1):
    client = OpenAI(base_url=f"http://{host}:{port}/v1",
                    api_key=api_key, )

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.1,
        stop=['\]\n\n', '\)\n\n', '# END!'],
        max_tokens=1024,
        extra_body={"include_stop_str_in_output": True, "repetition_penalty": repetition_penalty},
        n=n

    )
    return [choice.text for choice in completion.choices]

