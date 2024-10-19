# Medical Question Answering with Chain-of-Thought (CoT) Reasoning

This repository implements a **Chain-of-Thought (CoT)** reasoning approach for medical question answering using the [**PMC_LLaMA 13B** model](https://huggingface.co/axiong/PMC_LLaMA_13B). The system generates medical knowledge-based answers and evaluates its accuracy across several benchmark datasets.

This project is inspired by and refers to the work done in [**MedRAG**](https://github.com/Teddy-XiongGZ/MedRAG).

## Features

- **Chain-of-Thought Reasoning**: The system breaks down medical questions to generate coherent and logical answers.
- **Available Benchmarks**: Supports multiple datasets for medical question answering:
  - **bioasq.json**
  - **medmcqa.json**
  - **medqa.json**
  - **mmlu-med.json**
  - **pubmedqa.json**
- Efficient question answering using the **PMC_LLaMA 13B** model or the latest **MMed-Llama 3-8B**.
- Optimized for performance on CPU/GPU.
- Supports timeouts to handle large datasets and limit question processing time.
- Real-time accuracy evaluation during the question-answering process.

## Benchmarks

The reference benchmarks used for evaluation are from:
- **bioasq.json**
- **medmcqa.json**
- **medqa.json**
- **mmlu-med.json**
- **pubmedqa.json**

The datasets are automatically downloaded when you clone this repository.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### 1. Clone the Repository

First, clone this repository. The benchmark datasets will be downloaded automatically:

```bash
git clone https://github.com/superbabii/MedRAG-PMC-CoT.git
cd MedRAG-PMC-CoT
```

### 2. Install Dependencies

After cloning the repository, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all the required packages for running the code.

### 3. Install CUDA Toolkit and NVIDIA Drivers (Optional for GPU support)

If you plan to run the model on a GPU, you will need to install the NVIDIA CUDA toolkit and drivers:

```bash
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-driver-535
```

Make sure you have a compatible GPU before installing these packages.

### 4. Verify CUDA Installation (Optional)

After installing the CUDA toolkit and drivers, you can verify if CUDA is available with PyTorch by running the following command:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, then CUDA is available and ready for use with PyTorch. If `False`, check your installation and GPU compatibility.

## Available Models

The project supports different versions of the PMC-LLaMA model:

1. **PMC-LLaMA 13B**:
   - Model URL: [PMC_LLaMA 13B](https://huggingface.co/axiong/PMC_LLaMA_13B)
   - GitHub Repository: [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)

2. **Latest Version: MMed-Llama 3-8B**:
   - Model URL: [MMed-Llama 3-8B](https://huggingface.co/Henrychur/MMed-Llama-3-8B)
   - Additional Information: [PMC-LLaMA GitHub Repository](https://github.com/chaoyi-wu/PMC-LLaMA)

You can initialize the Chain-of-Thought reasoning model using either of these versions.

## Usage

The main script `main.py` includes the following steps for initializing and running the evaluation process:

### 1. Initialize the Chain-of-Thought (CoT) Reasoning Model

This project uses either the **PMC_LLaMA 13B** or the latest **MMed-Llama 3-8B** model for medical reasoning with **Chain-of-Thought reasoning** (not RAG). The model is initialized as follows:

```python
from src.medrag import MedRAG

# Initialize the MedRAG system with the desired model
cot = MedRAG(llm_name="Henrychur/MMed-Llama-3-8B", rag=False)
```

### 2. Running the Evaluation

The script processes the questions in the benchmark dataset and generates answers for each question using **Chain-of-Thought reasoning**. To limit the number of questions processed, you can modify this part of the code:

```python
# Limit to the first 1000 questions
all_questions = all_questions[:1000]
```

### 3. Timeout Handling

Each question is processed with a 30-second timeout limit to prevent long-running tasks:

```python
import signal

def handler(signum, frame):
    raise TimeoutError("Processing timed out.")

signal.signal(signal.SIGALRM, handler)
signal.alarm(30)  # Set alarm for 30 seconds
```

If the model takes too long to generate an answer, it will skip the question and move on to the next one.

### 4. Real-Time Accuracy

The script provides real-time feedback on the model’s accuracy as it processes the questions:

```python
print(f"Generated Answer (Raw): {generated_answer}")
print(f"Correct Answer: {correct_answer}")
print(f"Is Correct: {is_correct}")
print(f"Current Accuracy: {current_accuracy}%")
print(f"All Questions (Answered Questions): {total_questions} ({answered_questions})")
```

## Performance Comparison

Results show significant improvement over the published results from [**MIRAGE**](https://teddy-xionggz.github.io/MIRAGE/), especially across multiple benchmarks. Below is a comparison of the results:

| System               | LLM                | MMLU-Med | MedQA-US | MedMCQA | PubMedQA* | BioASQ-Y/N | Average |
|----------------------|--------------------|----------|----------|---------|-----------|------------|---------|
| **MIRAGE**            | PMC-LLaMA + CoT    | 52.16    | 44.38    | 46.55   | 55.80     | 63.11      | 52.40   |
| **This Result**    | PMC-LLaMA + CoT    | **59.80**| **53.23**| **51.85**| 43.55     | **63.75**  | **54.43** |

## License

This project is licensed under the MIT License.
```

This README covers installation, usage, and the addition of the new model, providing a complete guide for users to set up and run the medical question-answering system. Let me know if there’s anything else you’d like to add!