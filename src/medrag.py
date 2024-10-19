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

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_name,
            cache_dir=self.cache_dir,
            legacy=False
        )

        # Load the model using bf16 for optimized memory usage
        self.model = transformers.LlamaForCausalLM.from_pretrained(
            self.llm_name, 
            cache_dir=self.cache_dir, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Set max length to a smaller value for faster inference
        self.max_length = 2048
        
        # Ensure the tokenizer has a pad token if it doesn't already
        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad token, setting pad token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print confirmation that the model has been loaded on devices
        print(f"Model automatically loaded on appropriate devices using `device_map`.")

    def generate(self, prompt):
        # Simplified text generation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,  # Enable padding if batching
            truncation=True,  # Truncate to handle long prompts
            max_length=self.max_length
        )

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs['input_ids'],
                max_length=self.max_length,  # Reduce length for faster responses
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=self.model.config.pad_token_id
            )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def medrag_answer(self, question, options=None, save_dir=None):
        # Chain of Thoughts Configuration
        gpt_chain_of_thoughts = {
            "prompt_name": "gpt_chain_of_thoughts",
            "response_type": "MC",
            "prompt": question,
            "examples": [
                {
                    "question": question,
                    "answer": options
                }
            ]
        }
        
        if options:
            # Ensure options are sorted by key (e.g., A, B, C, D)
            sorted_keys = sorted(options.keys())
            options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted_keys])
        else:
            options_text = ''
        
        # Construct the prompt
        prompt = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
        
        # Generate the answer
        answer = self.generate(prompt).strip()

        # Optionally save the result
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            response_path = os.path.join(save_dir, "response.json")
            with open(response_path, 'w') as f:
                json.dump({"answer": answer, "gpt_chain_of_thoughts": gpt_chain_of_thoughts}, f, indent=4)
            print(f"Response saved to {response_path}")

        return answer
