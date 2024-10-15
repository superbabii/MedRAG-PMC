# import os
# import re
# import json
# import torch
# import transformers
# from transformers import AutoTokenizer
# import sys
# sys.path.append("src")
# from template import *

# class MedRAG:
#     def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, cache_dir=None):
#         self.llm_name = llm_name
#         self.rag = rag
#         self.cache_dir = cache_dir

#         # Loading templates
#         self.templates = {
#             "cot_system": general_cot_system,
#             "cot_prompt": general_cot,
#             "medrag_system": general_medrag_system,
#             "medrag_prompt": general_medrag,
#         }

#         self.max_length = 2048
#         self.context_length = 1024
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
#         self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')

#         self.model = transformers.pipeline(
#             "text-generation",
#             model=self.llm_name,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             model_kwargs={"cache_dir": self.cache_dir},
#         )

#         self.answer = self.medrag_answer

#     def generate(self, messages):
#         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         response = self.model(
#             prompt,
#             do_sample=False,
#             eos_token_id=self.tokenizer.eos_token_id,
#             pad_token_id=self.tokenizer.eos_token_id,
#             max_length=self.max_length,
#             truncation=True,
#         )
#         ans = response[0]["generated_text"][len(prompt):]
#         return ans

#     def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir=None, **kwargs):
#         options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options)]) if options else ''

#         # Generate answers
#         prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_text)
#         messages = [
#             {"role": "system", "content": self.templates["cot_system"]},
#             {"role": "user", "content": prompt_cot},
#         ]

#         answer = self.generate(messages).strip()

#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#             with open(os.path.join(save_dir, "response.json"), 'w') as f:
#                 json.dump(answer, f, indent=4)

#         return answer

#     def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path=None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
#         options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options)]) if options else ''
#         question_prompt = f"Here is the question:\n{question}\n\n{options_text}"

#         context, qa_cache = '', []
#         if qa_cache_path and os.path.exists(qa_cache_path):
#             with open(qa_cache_path, 'r') as f:
#                 qa_cache = json.load(f)[:n_rounds]
#             context = qa_cache[-1] if qa_cache else ''
#             n_rounds -= len(qa_cache)

#         saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]
#         last_context = None

#         for i in range(n_rounds + 3):
#             # Preparing messages for each round
#             if i < n_rounds:
#                 user_prompt = f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_ask'].format(n_queries)}" if context else f"{question_prompt}\n\n{self.templates['follow_up_ask'].format(n_queries)}"
#             elif context != last_context:
#                 user_prompt = f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_answer']}"
#             else:
#                 continue

#             messages = [{"role": "system", "content": self.templates["i_medrag_system"]}, {"role": "user", "content": user_prompt}]
#             saved_messages.append(messages[-1])

#             # Generate response
#             last_context = context
#             last_content = self.generate(messages).strip()
#             response_message = {"role": "assistant", "content": last_content}
#             saved_messages.append(response_message)

#             # Save progress
#             if save_path:
#                 with open(save_path, 'w') as f:
#                     json.dump([m for m in saved_messages], f, indent=4)

#             # Extract answer or queries
#             if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
#                 messages.append({"role": "user", "content": "Output the answer in JSON."})
#                 return self.generate(messages), saved_messages

#             if "## Queries" in last_content:
#                 queries = re.findall(r"\d+\.\s*(.*?)\n", last_content)
#                 for query in queries:
#                     try:
#                         context += f"\n\nQuery: {query}\nAnswer: {self.medrag_answer(query, k=k, rrf_k=rrf_k)}"
#                     except Exception as e:
#                         continue
#                 if qa_cache_path:
#                     with open(qa_cache_path, 'w') as f:
#                         json.dump(qa_cache, f, indent=4)

#         return saved_messages[-1]["content"], saved_messages


import os
import json
import torch
import transformers
from transformers import AutoTokenizer

class MedRAG:
    def __init__(self, llm_name="axiong/PMC_LLaMA_13B", rag=True, cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.cache_dir = cache_dir

        # Simplified to load model once and reuse
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
        self.model = transformers.LlamaForCausalLM.from_pretrained(self.llm_name, cache_dir=self.cache_dir)

        self.max_length = 1000  # Shorter max length for faster inference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # Move the model to GPU if available

    def generate(self, prompt):
        # Simplified text generation
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_length=self.max_length,  # Reduce length for faster responses
                do_sample=True,
                top_k=50,
                temperature=0.7  # Add some randomness
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def medrag_answer(self, question, options=None, save_dir=None):
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options)]) if options else ''
        prompt = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
        
        # Generate the answer
        answer = self.generate(prompt).strip()

        # Optionally save the result
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answer, f, indent=4)

        return answer
