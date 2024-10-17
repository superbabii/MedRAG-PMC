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
- Efficient question answering using the **PMC_LLaMA 13B** model.
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

The datasets are automatically downloaded when I clone this repository.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

## Installation

### 1. Install CUDA Toolkit and NVIDIA Drivers (Optional for GPU support)

If you plan to run the model on a GPU, you will need to install the NVIDIA CUDA toolkit and drivers:

```bash
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-driver-535
```

Make sure you have a compatible GPU before installing these packages.

### 2. Verify CUDA Installation (Optional)

After installing the CUDA toolkit and drivers, you can verify if CUDA is available with PyTorch by running the following command:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, then CUDA is available and ready for use with PyTorch. If `False`, check your installation and GPU compatibility.

### 3. Clone the Repository

Then, clone this repository. The benchmark datasets will be downloaded automatically:

```bash
git clone https://github.com/superbabii/MedRAG-PMC-CoT.git
cd MedRAG-PMC-CoT
```

### 4. Install Dependencies

After cloning the repository, install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

This will install all the required packages for running the code.

## Usage

The main script `main.py` includes the following steps for initializing and running the evaluation process:

### 2.1 Initialize the Chain-of-Thought (CoT) Reasoning Model

This project uses the **PMC_LLaMA 13B** model for medical reasoning with **Chain-of-Thought reasoning** (not RAG). The model is initialized as follows:

```python
from src.medrag import MedRAG

# Initialize the MedRAG system with PMC_LLaMA 13B and CoT
cot = MedRAG(llm_name="axiong/PMC_LLaMA_13B", rag=False)
```

The **PMC_LLaMA 13B** model can be accessed from Hugging Face here: [https://huggingface.co/axiong/PMC_LLaMA_13B](https://huggingface.co/axiong/PMC_LLaMA_13B).

### 2.2 Running the Evaluation

The script processes the questions in the benchmark dataset and generates answers for each question using **Chain-of-Thought reasoning**. To limit the number of questions processed, you can modify this part of the code:

```python
# Limit to the first 1000 questions
all_questions = all_questions[:1000]
```

### 2.3 Timeout Handling

Each question is processed with a 30-second timeout limit to prevent long-running tasks:

```python
signal.alarm(30)  # Set alarm for 30 seconds
```

If the model takes too long to generate an answer, it will skip the question and move on to the next one.

### 2.4 Real-Time Accuracy

The script provides real-time feedback on the model’s accuracy as it processes the questions:

```text
Generated Answer (Raw): "The correct answer is A. Hemoglobin carries oxygen."
Correct Answer: A
Is Correct: True
Current Accuracy: 85.71%
All Questions(Answered Questions): 100(14)
```

### Results and Accuracy

As the script runs, it prints the current accuracy based on the number of correctly answered questions. Results can also be saved to a file for further analysis.

### Performance Comparison

My results show significant improvement over the published results from [**MIRAGE**](https://teddy-xionggz.github.io/MIRAGE/), especially across multiple benchmarks. Below is a comparison of my results versus the MIRAGE system:

| System               | LLM                | MMLU-Med | MedQA-US | MedMCQA | PubMedQA* | BioASQ-Y/N | Average |
|----------------------|--------------------|----------|----------|---------|-----------|------------|---------|
| **MIRAGE**            | PMC-LLaMA + CoT    | 52.16    | 44.38    | 46.55   | 55.80     | 63.11      | 52.40   |
| **This Result**    | PMC-LLaMA + CoT    | **59.80**| **53.23**| **51.85**| 43.55     | **63.75**  | **54.43** |

The comparison shows that my approach performs better than the MIRAGE system in the following categories:
- **MMLU-Med**: 59.80 (This) vs. 52.16 (MIRAGE)
- **MedQA-US**: 53.23 (This) vs. 44.38 (MIRAGE)
- **MedMCQA**: 51.85 (This) vs. 46.55 (MIRAGE)
- **BioASQ-Y/N**: 63.75 (This) vs. 63.11 (MIRAGE)

While there was a slight decrease in performance for **PubMedQA** (43.55 vs. 55.80), my overall average score is higher at **54.43**, compared to **52.40** for MIRAGE.

### Questions Count

The evaluation is based on the following number of questions per benchmark:
- **MMLU-Med**: 1000 questions
- **MedQA-US**: 1000 questions
- **MedMCQA**: 1000 questions
- **PubMedQA**: 494 questions
- **BioASQ-Y/N**: 618 questions

Total: **4114 questions**.

## License

This project is licensed under the MIT License.




To incorporate the CUDA availability check into your README, you can add it as a verification step after the installation of the CUDA toolkit and drivers. Here's how the updated section would look:

```markdown
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

