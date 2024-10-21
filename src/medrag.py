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
import random
import torch
import transformers
from transformers import AutoTokenizer
from collections import Counter

# Refined system prompt with explicit instructions for reasoning
system_prompt = """You are a highly knowledgeable medical professional. 
For each medical question with multiple-choice answers, carefully consider each option and think through it step by step.
- For each choice, explain why it could be correct or incorrect, including any relevant medical knowledge.
- Double-check your reasoning for consistency and accuracy before deciding.
- Clearly state your final answer and specify the corresponding letter choice.

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
(Provide a detailed chain of thought explanation, checking each option thoroughly)
Therefore, the answer is [final model answer (e.g., A, B, C, or D)]."""

def create_query(item, shuffled_options=None):
    # Use shuffled options if provided, otherwise fall back to original options
    options = shuffled_options if shuffled_options else item["options"]
    
    # Enhanced query format with additional context
    query = f"""## Question: Given the following medical scenario, analyze each option carefully:
{item["question"]}
A. {options["A"]}
B. {options["B"]}
C. {options["C"]}
D. {options["D"]}"""
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

# examples = {
#     "anatomy-000": {
#     "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
#     "options": {
#         "A": "paralysis of the facial muscles.",
#         "B": "paralysis of the facial muscles and loss of taste.",
#         "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
#         "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
#     },
#     "answer": "A"
#     },
#     "anatomy-001": {
#     "question": "A \"dished face\" profile is often associated with",
#     "options": {
#         "A": "a protruding mandible due to reactivation of the condylar cartilage by acromegaly.",
#         "B": "a recessive maxilla due to failure of elongation of the cranial base.",
#         "C": "an enlarged frontal bone due to hydrocephaly.",
#         "D": "defective development of the maxillary air sinus."
#     },
#     "answer": "B"
#     },
#     "anatomy-002": {
#     "question": "Which of the following best describes the structure that collects urine in the body?",
#     "options": {
#         "A": "Bladder",
#         "B": "Kidney",
#         "C": "Ureter",
#         "D": "Urethra"
#     },
#     "answer": "A"
#     },
#     "anatomy-003": {
#     "question": "Which of the following structures is derived from ectomesenchyme?",
#     "options": {
#         "A": "Motor neurons",
#         "B": "Skeletal muscles",
#         "C": "Melanocytes",
#         "D": "Sweat glands"
#     },
#     "answer": "C"
#     },
#     "anatomy-004": {
#     "question": "Which of the following describes the cluster of blood capillaries found in each nephron in the kidney?",
#     "options": {
#         "A": "Afferent arteriole",
#         "B": "Glomerulus",
#         "C": "Loop of Henle",
#         "D": "Renal pelvis"
#     },
#     "answer": "B"
#     },
#     "anatomy-005": {
#     "question": "A patient suffers a broken neck with damage to the spinal cord at the level of the sixth cervical vertebra.",
#     "options": {
#         "A": "They will be unable to breathe without life support.",
#         "B": "They will only be able to breathe quietly.",
#         "C": "It is impossible to predict an effect on breathing.",
#         "D": "Breathing will be unaffected."
#     },
#     "answer": "B"
#     }
# }

examples = [
    {
      "question": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral",
      "options": {
        "A": "paralysis of the facial muscles.",
        "B": "paralysis of the facial muscles and loss of taste.",
        "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
        "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
      },
      "answer": "A"
    },
    {
      "question": "A \"dished face\" profile is often associated with",
      "options": {
        "A": "a protruding mandible due to reactivation of the condylar cartilage by acromegaly.",
        "B": "a recessive maxilla due to failure of elongation of the cranial base.",
        "C": "an enlarged frontal bone due to hydrocephaly.",
        "D": "defective development of the maxillary air sinus."
      },
      "answer": "B"
    },
    {
      "question": "Which of the following best describes the structure that collects urine in the body?",
      "options": {
        "A": "Bladder",
        "B": "Kidney",
        "C": "Ureter",
        "D": "Urethra"
      },
      "answer": "A"
    },
]

def build_few_shot_prompt(system_prompt, question, examples, include_cot=True):
    # Builds a few-shot prompt with examples for more effective learning
    messages = [{"role": "system", "content": system_prompt}]
    
    for elem in examples:
        messages.append({"role": "user", "content": create_query(elem)})
        if include_cot:
            messages.append({"role": "assistant", "content": format_answer(elem["cot"], elem["answer"])})
        else:
            answer_string = f"""## Answer\nTherefore, the answer is {elem["answer"]}"""
            messages.append({"role": "assistant", "content": answer_string})
    
    messages.append({"role": "user", "content": create_query(question)})
    return messages 

def shuffle_option_labels(answer_options):
    options = list(answer_options.items())
    random.shuffle(options)
    labels = [chr(i) for i in range(ord('A'), ord('A') + len(options))]
    shuffled_options_dict = {label: option_text for label, (original_label, option_text) in zip(labels, options)}
    original_mapping = {label: original_label for label, (original_label, _) in zip(labels, options)}
    return shuffled_options_dict, original_mapping

def extract_answer_choice(generated_answer, valid_choices=("A", "B", "C", "D")):
    # Only consider options within the valid range (A-D)
    pattern = rf"OPTION ([{''.join(valid_choices)}]) IS CORRECT"
    match = re.search(pattern, generated_answer, re.IGNORECASE)
    if match:
        choice = match.group(1).upper()
        if choice in valid_choices:
            return choice
    return None  # Return None if no valid choice is found

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
                    top_p=1.0,
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
        # Start from the first occurrence of "## Answer" and filter out placeholder text
        start_index = raw_output.find("## Answer")
        if start_index != -1:
            filtered_output = raw_output[start_index:].strip()
            # Remove placeholder instructions like "(Provide a detailed chain of thought explanation)"
            filtered_output = re.sub(r"\(Provide a detailed chain of thought explanation\)", "", filtered_output)
            filtered_output = re.sub(r"Therefore, the answer is \[final model answer \(e\.g\., A, B, C, or D\)\]\.", "", filtered_output)
            return filtered_output.strip()
        else:
            # If "## Answer" is not found, return the original output
            return raw_output

    def medrag_answer(self, question_data, save_dir=None, shuffle=True, num_shuffles=5):
        answer_counts = Counter()
        shuffle_results = []

        for _ in range(num_shuffles):
            # Shuffle options and get the mapping to original labels
            shuffled_options, original_mapping = shuffle_option_labels(question_data["options"]) if shuffle else (question_data["options"], {label: label for label in question_data["options"]})
            
            # Generate the prompt with the shuffled options
            prompt = build_zero_shot_prompt(system_prompt, {"question": question_data["question"], "options": shuffled_options})
            # prompt = build_few_shot_prompt(system_prompt, {"question": question_data["question"], "options": shuffled_options}, examples, include_cot=False)            
            raw_answer = self.generate(prompt)

            # Extract the option letter from the raw answer
            extracted_choice = extract_answer_choice(raw_answer)

            if extracted_choice and extracted_choice in shuffled_options:
                # Map the shuffled choice back to the original label
                original_label = original_mapping[extracted_choice]
                answer_counts[original_label] += 1
                shuffle_results.append((shuffled_options, original_label, raw_answer))
            else:
                # If the answer is not valid, log as "Unknown"
                shuffle_results.append((shuffled_options, "Unknown", raw_answer))

        # Determine the most common answers
        most_common_answers = answer_counts.most_common()
        highest_frequency = most_common_answers[0][1] if most_common_answers else 0

        # Find all answers with the highest frequency
        tied_answers = [answer for answer, freq in most_common_answers if freq == highest_frequency]

        # If there's a tie, prioritize based on original order (A, B, C, D)
        if len(tied_answers) > 1:
            # Sort tied answers based on their original order in question_data["options"]
            most_common_answer = min(tied_answers, key=lambda x: list(question_data["options"].keys()).index(x))
        else:
            most_common_answer = tied_answers[0] if tied_answers else "Unknown"

        return {
            "final_answer": most_common_answer,  # This will now be one of 'A', 'B', 'C', or 'D'
            "frequency": highest_frequency,
            "details": shuffle_results
        }

