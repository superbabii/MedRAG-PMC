# import os
# import re
# import json
# import torch
# import transformers
# from transformers import AutoTokenizer
# import sys
# sys.path.append("src")
# from template import *
# from liquid import Template

# general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

# general_cot = Template('''
# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json:
# ''')

# general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

# general_medrag = Template('''
# Here are the relevant documents:
# {{context}}

# Here is the question:
# {{question}}

# Here are the potential choices:
# {{options}}

# Please think step-by-step and generate your output in json:
# ''')

# class MedRAG:
#     def __init__(self, llm_name="axiong/PMC_LLaMA_13B", rag=True, cache_dir=None):
#         self.llm_name = llm_name
#         self.rag = rag
#         self.cache_dir = cache_dir
        
#         self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
#                     "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}

#         # Load the tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.llm_name,
#             cache_dir=self.cache_dir,
#             legacy=False
#         )

#         # Load the model using bf16 for optimized memory usage
#         self.model = transformers.LlamaForCausalLM.from_pretrained(
#             self.llm_name, 
#             cache_dir=self.cache_dir, 
#             torch_dtype=torch.bfloat16,
#             device_map="auto"
#         )

#         # Set max length to a smaller value for faster inference
#         self.max_length = 2048
        
#         # Ensure the tokenizer has a pad token if it doesn't already
#         if self.tokenizer.pad_token is None:
#             print("Tokenizer has no pad token, setting pad token to eos_token.")
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#     def generate(self, prompt):
#         # Convert list of dictionaries to a single string if needed
#         if isinstance(prompt, list):
#             prompt = ' '.join([msg['content'] for msg in prompt if 'content' in msg])

#         # Simplified text generation
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             add_special_tokens=False,
#             padding=True,  # Enable padding if batching
#             truncation=True,  # Truncate to handle long prompts
#             max_length=self.max_length
#         )

#         if self.llm_name == "Henrychur/MMed-Llama-3-8B":
#             # Move inputs to the correct device
#             inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#             # Generate text
#             with torch.no_grad():
#                 generated_ids = self.model.generate(
#                     inputs['input_ids'],
#                     attention_mask=inputs['attention_mask'],  # Explicitly set attention mask
#                     max_length=self.max_length,  # Reduce length for faster responses
#                     do_sample=True,
#                     top_k=50,
#                     temperature=0.7,
#                     pad_token_id=self.model.config.pad_token_id
#                 )
#         elif self.llm_name == "axiong/PMC_LLaMA_13B":
#             # No need to move inputs to the device manually with device_map="auto"
#             with torch.no_grad():
#                 generated_ids = self.model.generate(
#                     inputs['input_ids'],
#                     max_length=self.max_length,  # Reduce length for faster responses
#                     do_sample=True,
#                     top_k=50,
#                     temperature=0.7,
#                     pad_token_id=self.model.config.pad_token_id
#                 )

#         return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


#     def medrag_answer(self, question, options=None, save_dir=None):
#         if options is not None:
#             options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
#         else:
#             options = ''
#         prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
#         messages = [
#             {"role": "system", "content": self.templates["cot_system"]},
#             {"role": "user", "content": prompt_cot}
#         ]
#         answers = []
#         ans = self.generate(messages)
#         answers.append(re.sub("\s+", " ", ans))

#         # Optionally save the result
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             response_path = os.path.join(save_dir, "response.json")
#             with open(response_path, 'w') as f:
#                 json.dump({"answer": answers}, f, indent=4)
#             print(f"Response saved to {response_path}")

#         return answers[0] if len(answers)==1 else answers


import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer

# Refined system prompt with explicit instructions for reasoning
system_prompt = """You are a highly knowledgeable medical professional. 
For each medical question with multiple-choice answers, think through each option carefully.
Explain the reasoning step by step, considering why each choice is correct or incorrect.
Conclude with a final answer and specify the corresponding letter choice.

Follow this format:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
(Provide a detailed chain of thought explanation)
Therefore, the answer is [final model answer (e.g., A, B, C, or D)]."""

def create_query(item):
    # Enhanced query format with additional context
    query = f"""## Question: Given the following medical scenario, analyze each option carefully:
{item["question"]}
A. {item["options"]["A"]}
B. {item["options"]["B"]}
C. {item["options"]["C"]}
D. {item["options"]["D"]}"""
    return query

def format_answer(cot, answer):
    # Formats the output as specified in the system prompt
    return f"""## Answer
{cot}
Therefore, the answer is {answer}"""

def build_zero_shot_prompt(system_prompt, question):
    # Builds a zero-shot prompt with only the system instructions and the current question
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": create_query(question)}]
    return messages

def build_few_shot_prompt(system_prompt, question, examples, include_cot=True):
    # Builds a few-shot prompt with examples for more effective learning
    messages = [{"role": "system", "content": system_prompt}]
    
    for elem in examples:
        messages.append({"role": "user", "content": create_query(elem)})
        if include_cot:
            messages.append({"role": "assistant", "content": format_answer(elem["cot"], elem["answer_idx"])})
        else:
            answer_string = f"""## Answer\nTherefore, the answer is {elem["answer_idx"]}"""
            messages.append({"role": "assistant", "content": answer_string})
    
    messages.append({"role": "user", "content": create_query(question)})
    return messages 

class MedRAG:
    def __init__(self, llm_name="axiong/PMC_LLaMA_13B", rag=True, cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.cache_dir = cache_dir

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            cache_dir=self.cache_dir,
            legacy=False
        )

        # Load the model with optimized memory usage using bf16
        self.model = transformers.LlamaForCausalLM.from_pretrained(
            self.llm_name, 
            cache_dir=self.cache_dir, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Set max length for inputs to avoid excessively long prompts
        self.max_length = 2048
        
        # Ensure the tokenizer has a pad token, otherwise set it to eos_token
        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad token, setting pad token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt):
        # Convert list of messages to a single prompt string
        if isinstance(prompt, list):
            prompt = ' '.join([msg['content'] for msg in prompt if 'content' in msg])
        
        # Tokenize the input prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,  # Enable padding
            truncation=True,  # Truncate if too long
            max_length=self.max_length
        ).to(self.model.device)

        # Generate the response
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['input_ids'],
                    max_length=self.max_length,  # Limit response length
                    do_sample=True,
                    top_k=50,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode the generated response
            raw_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            
            # Post-process the output to filter out unwanted content
            processed_output = self._filter_output(raw_output)
            return processed_output
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Generation failed."

    def _filter_output(self, raw_output):
        # Remove prompt instructions and unnecessary content from the generated output
        start_index = raw_output.find("## Answer")
        if start_index != -1:
            return raw_output[start_index:].strip()
        else:
            # If "## Answer" is not found, return the original output
            return raw_output

    def medrag_answer(self, question, save_dir=None):
        # Build the prompt for zero-shot learning
        prompt = build_zero_shot_prompt(system_prompt, question)
        
        # Generate the answer
        answer = self.generate(prompt)

        # Optionally save the result
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            response_path = os.path.join(save_dir, "response.json")
            with open(response_path, 'w') as f:
                json.dump({"answer": answer}, f, indent=4)
            print(f"Response saved to {response_path}")

        return answer