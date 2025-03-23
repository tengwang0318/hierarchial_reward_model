def construct_policy_model_prompt_for_PRM_HRM(question: str, previous_reasoning: str = None) -> str:
    if not previous_reasoning:
        template = f"""You are an expert of Math and need to solve the following question and return the answer.

Question:
{question}


Let's analyze this step by step.

Begin each step with '# Step X' to clearly indicate the entire reasoning step.
After you finish thinking, you need to output the answer again! 
The answer should start with '# Answer', followed by two line breaks and the final response.
Just provide the answer value without any descriptive text at the end. 
And the answer ends with '# END!'
Below is a correct example of the expected output format:
-----------------
Question: 1+2+3 = ?

# Step 1
solve 1 + 2 = 3,

# Step 2
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

Begin each step with '# Step X' to clearly indicate the entire reasoning step.
After you finish thinking, you need to output the answer again! 
The answer should start with '# Answer', followed by two line breaks and the final response.
Just provide the answer value without any descriptive text at the end. 
And the answer ends with '# END!'
Below is a correct example of the expected output format:
-----------------
Question: 1+2+3 = ?

# Step 1
solve 1 + 2 = 3,

# Step 2
Then, 3 + 3 = 6.

# Answer  

6
# END!
-----------------
{previous_reasoning}
"""
    return template

def construct_PRM_HRM_prompt_v2(question, previous_steps, current_step):
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
    # print("----\nPRM_REWARD_MODEL_PROMPT:")
    # print(template)
    # print("-----finish prm prompt")
    return template
