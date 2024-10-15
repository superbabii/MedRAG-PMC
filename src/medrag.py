import os
import re
import json
import torch
import transformers
from transformers import AutoTokenizer
from template import *


class MedRAG:
    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.cache_dir = cache_dir

        # Loading templates
        self.templates = {
            "cot_system": general_cot_system,
            "cot_prompt": general_cot,
            "medrag_system": general_medrag_system,
            "medrag_prompt": general_medrag,
        }

        self.max_length = 2048
        self.context_length = 1024
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
        self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')

        self.model = transformers.pipeline(
            "text-generation",
            model=self.llm_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            model_kwargs={"cache_dir": self.cache_dir},
        )

        self.answer = self.medrag_answer

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = self.model(
            prompt,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            truncation=True,
        )
        ans = response[0]["generated_text"][len(prompt):]
        return ans

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir=None, **kwargs):
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options)]) if options else ''

        # Generate answers
        prompt_cot = self.templates["cot_prompt"].render(question=question, options=options_text)
        messages = [
            {"role": "system", "content": self.templates["cot_system"]},
            {"role": "user", "content": prompt_cot},
        ]

        answer = self.generate(messages).strip()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answer, f, indent=4)

        return answer

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path=None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options)]) if options else ''
        question_prompt = f"Here is the question:\n{question}\n\n{options_text}"

        context, qa_cache = '', []
        if qa_cache_path and os.path.exists(qa_cache_path):
            with open(qa_cache_path, 'r') as f:
                qa_cache = json.load(f)[:n_rounds]
            context = qa_cache[-1] if qa_cache else ''
            n_rounds -= len(qa_cache)

        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]
        last_context = None

        for i in range(n_rounds + 3):
            # Preparing messages for each round
            if i < n_rounds:
                user_prompt = f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_ask'].format(n_queries)}" if context else f"{question_prompt}\n\n{self.templates['follow_up_ask'].format(n_queries)}"
            elif context != last_context:
                user_prompt = f"{context}\n\n{question_prompt}\n\n{self.templates['follow_up_answer']}"
            else:
                continue

            messages = [{"role": "system", "content": self.templates["i_medrag_system"]}, {"role": "user", "content": user_prompt}]
            saved_messages.append(messages[-1])

            # Generate response
            last_context = context
            last_content = self.generate(messages).strip()
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)

            # Save progress
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([m for m in saved_messages], f, indent=4)

            # Extract answer or queries
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append({"role": "user", "content": "Output the answer in JSON."})
                return self.generate(messages), saved_messages

            if "## Queries" in last_content:
                queries = re.findall(r"\d+\.\s*(.*?)\n", last_content)
                for query in queries:
                    try:
                        context += f"\n\nQuery: {query}\nAnswer: {self.medrag_answer(query, k=k, rrf_k=rrf_k)}"
                    except Exception as e:
                        continue
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)

        return saved_messages[-1]["content"], saved_messages

