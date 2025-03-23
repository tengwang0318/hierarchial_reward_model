def construct_policy_model_prompt_for_ORM(question):
    # This is the policy model's prompt for ORM task
    template = f"""You are an expert of Math and need to solve the following question and return the answer.
    
Question:
{question}

Let's analyze this step by step.

After you finish thinking, you need to output the answer again! 
The answer should start with '# Answer', followed by two line breaks and the final response.
Just provide the answer value without any descriptive text at the end. 
And the answer ends with '# END!'
Below is a correct example of the expected output format:
-----------------
Question: 1+2+3 = ?

Firstly, solve 1 + 2 = 3,
Then, 3 + 3 = 6.

# Answer  

6
# END!
-----------------
"""

    return template


def construct_policy_model_prompt_for_PRM_HRM(question: str, previous_reasoning: str = None) -> str:
    # This is the policy model's prompt for PRM and HRM task
    if not previous_reasoning:
        template = f"""You are an expert of Math and need to solve the following question and return the answer.
        
Question:
{question}


Let's analyze this step by step.

After you finish thinking, you need to output the answer again! 
The answer should start with '# Answer', followed by two line breaks and the final response.
Just provide the answer value without any descriptive text at the end. 
And the answer ends with '# END!'
Below is a correct example of the expected output format:
-----------------
Question: 1+2+3 = ?

Firstly, solve 1 + 2 = 3,
Then, 3 + 3 = 6.

# Answer  

6
# END!
-----------------
"""
    else:
        template = f"""You are an expert of Math and need to solve the following question and return the answer.
        
Question:
{question}


Let's analyze this step by step.

After you finish thinking, you need to output the answer again! 
The answer should start with '# Answer', followed by two line breaks and the final response.
Just provide the answer value without any descriptive text at the end. 
And the answer ends with '# END!'
Below is a correct example of the expected output format:
-----------------
Question: 1+2+3 = ?

Firstly, solve 1 + 2 = 3,
Then, 3 + 3 = 6.

# Answer  

6
# END!
-----------------
{previous_reasoning}
"""
    return template


def construct_ORM_prompt(question, answer):
    # ORM prompt (RM)
    template = f"""Question is as follows:
{question}

The answer is as follows:
{answer}
"""
    return template


def construct_PRM_HRM_prompt_v2(question, previous_steps, current_step):
    # HRM and PRM prompt (RM) when calculating the score
    if previous_steps:
        template = f"""Question:
{question}

Let's break it down step by step!

Previous reasoning:
{previous_steps}

Now, let's focus on the current step:
{current_step}"""


    else:
        template = f"""Question:
{question}

Let's break it down step by step!

Now, let's focus on the current step:
{current_step}
"""
    return template


def construct_PRM_HRM_prompt(question, answer):
    # PRM and HRM prompt when construct training dataset for PRM800K dataset
    placeholder = " /qwerdf12344567"
    len_placeholder = len(placeholder)
    answer = answer[len_placeholder:]
    answer_slices = answer.split(placeholder)
    if len(answer_slices) == 1:
        current_step = answer_slices[0]
        previous_steps = None
    else:
        current_step = answer_slices[-1]
        previous_steps = "\n\n".join(answer_slices[:-1])
    if previous_steps:
        template = f"""Question:
{question}

Let's break it down step by step!

Previous reasoning:
{previous_steps}

Now, let's focus on the current step:
{current_step}"""


    else:
        template = f"""Question:
{question}

Let's break it down step by step!

Now, let's focus on the current step:
{current_step}
"""
    return template
