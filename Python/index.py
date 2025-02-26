'''
Author: yueshengqi
Date: 2025-02-21 22:12:15
LastEditors: Do not edit
LastEditTime: 2025-02-25 21:30:47
Description: 
FilePath: \LLM\Python\index.py
'''
import os
from huggingface_hub import InferenceClient

## You need a token from https://hf.co/settings/tokens, ensure that you select 'read' as the token type. If you run this on Google Colab, you can set it up in the "settings" tab under "secrets". Make sure to call it "HF_TOKEN"
os.environ["HF_TOKEN"]="xxxx"

# client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")
# if the outputs for next cells are wrong, the free model may be overloaded. You can also use this public endpoint that contains Llama-3.2-3B-Instruct
client = InferenceClient("https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud")

# output = client.text_generation(
#     "The capital of france is",
#     max_new_tokens=100,
# )

# print(output)

prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The capital of France is<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print(output)

#https://huggingface.co/agents-course/notebooks/blob/main/dummy_agent_library.ipynb
#https://huggingface.co/learn/agents-course/unit1/dummy-agent-library