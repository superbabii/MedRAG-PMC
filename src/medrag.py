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
        self.max_length = 150
        
        # Check if CUDA is available (for other use cases, but no need to move model manually)
        if torch.cuda.is_available():
            print("CUDA is available. Model is automatically moved by device_map.")
        else:
            print("CUDA not available. Using CPU.")

        # Ensure the tokenizer has a pad token if it doesn't already
        if self.tokenizer.pad_token is None:
            print("Tokenizer has no pad token, setting pad token to eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Print confirmation that the model has been loaded on devices
        print(f"Model automatically loaded on appropriate devices using `device_map`.")

    def _set_pad_token(self):
        # Check if tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            # Add a pad token
            special_tokens = {'pad_token': '[PAD]'}
            self.tokenizer.add_special_tokens(special_tokens)
            print("Added [PAD] token to the tokenizer.")

            # Resize model embeddings to accommodate the new pad token
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Set pad_token_id in the model configuration
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Also set it in the generation config
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Verify pad_token_id is positive
        if self.model.config.pad_token_id < 0:
            raise ValueError(f"Invalid pad_token_id: {self.model.config.pad_token_id}. It should be a positive integer.")
        else:
            print("pad_token_id is valid and positive.")
            
        # Check if pad_token_id is the same as eos_token_id
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            print("Warning: `pad_token_id` is set to the same value as `eos_token_id`. Consider setting a distinct pad token.")
            # Optionally, set a distinct pad token if needed
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({'pad_token': "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

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

        if self.llm_name == "Henrychur/MMed-Llama-3-8B":
            # Move inputs to the correct device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Explicitly set attention mask
                    max_length=self.max_length,  # Reduce length for faster responses
                    do_sample=True,
                    top_k=50,
                    temperature=0.7,
                    pad_token_id=self.model.config.pad_token_id
                )
        elif self.llm_name == "axiong/PMC_LLaMA_13B":
            # No need to move inputs to the device manually with device_map="auto"
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
        if options:
            # Ensure options are sorted by key (e.g., A, B, C, D)
            sorted_keys = sorted(options.keys())
            options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted_keys])
        else:
            options_text = ''
        prompt = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
        
        # Generate the answer
        answer = self.generate(prompt).strip()

        # Optionally save the result
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            response_path = os.path.join(save_dir, "response.json")
            with open(response_path, 'w') as f:
                json.dump({"answer": answer}, f, indent=4)
            print(f"Response saved to {response_path}")

        return answer
