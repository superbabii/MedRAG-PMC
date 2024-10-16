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

        self.max_length = 2048  # Shorter max length for faster inference
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
