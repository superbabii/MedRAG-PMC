import json
import re
import random
from src.medrag import MedRAG

# Load the benchmark JSON file
with open('bioasq.json', 'r') as f:
    benchmark_data = json.load(f)

# Get all questions
all_questions = list(benchmark_data.items())

# Limit to the first 200 questions
all_questions = all_questions[:800]
# all_questions = all_questions[600:1000]
# all_questions = all_questions[400:600]
# all_questions = all_questions[600:800]
# all_questions = all_questions[800:1000]

# Get random questions
# random_questions = random.sample(list(benchmark_data.items()), 5)

# Initialize the MedRAG system
cot = MedRAG(llm_name="axiong/PMC_LLaMA_13B", rag=False)

# Store the results of comparisons
results = []
correct_count = 0
total_questions = len(all_questions)  # Get the total number of questions

# Function to extract the answer choice
def extract_answer_choice(generated_answer):
    # Look for "OPTION X IS CORRECT" or "ANSWER IS X"
    match = re.search(r"OPTION ([A-D]) IS CORRECT", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Return the extracted option (A, B, C, or D)
    # As a fallback, look for "ANSWER IS X"
    match = re.search(r"ANSWER IS ([A-D])", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Look for "Answer: X"
    match = re.search(r"Answer:\s*([A-D])", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Return the extracted option (A, B, C, or D)
    return None  # If no valid option is found, return None

# Iterate over each question and get the generated answer
for question_id, question_data in all_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    # Use MedRAG to generate the answer
    generated_answer = cot.medrag_answer(question=question, options=options)
    
    print(f"Generated Answer (Raw): {generated_answer}")
    
    # Extract the generated answer choice
    generated_choice = extract_answer_choice(generated_answer)

    if not generated_choice:
        print(f"No valid answer choice extracted for question ID: {question_id}")
        continue

    # Compare the generated answer with the correct one
    is_correct = correct_answer == generated_choice
    if is_correct:
        correct_count += 1

    result = {
        'question_id': question_id,
        'question': question,
        'correct_answer': correct_answer,
        'generated_answer': generated_choice,
        'is_correct': is_correct
    }
    results.append(result)

# Print the results of the comparison
for result in results:
    print(f"Question: {result['question']}")
    print(f"Correct Answer: {result['correct_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print(f"Is Correct: {result['is_correct']}")
    print('-' * 50)

# Print the number of all questions
print(f"Total number of questions: {total_questions}")

# Calculate accuracy
accuracy = correct_count / total_questions * 100
print(f"Accuracy: {accuracy:.2f}%")
