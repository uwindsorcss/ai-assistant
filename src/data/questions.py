"""Script to take questions from the text file and put them in the jsonc file."""
import os
import random
import commentjson

jsonc_path = "./data/finetune/questions.jsonc"
text_file_path = "./data/finetune/questions.txt"

# Check if the JSONC file exists
if not os.path.exists(jsonc_path):
    print(f"Error: The file {jsonc_path} does not exist.")
    exit()

# Read and parse the existing JSONC file
with open(jsonc_path, 'r', encoding='utf-8') as jsonc_file:
    try:
        data = commentjson.load(jsonc_file)
    except Exception as e:
        print(f"Error parsing JSONC file: {e}")
        exit()

# Ensure 'questions' key exists and is a list
questions = data.get('questions', [])
if not isinstance(questions, list):
    print("Error: 'questions' should be a list in the JSONC file.")
    exit()

# Read new questions from the text file
if not os.path.exists(text_file_path):
    print(f"Error: The file {text_file_path} does not exist.")
    exit()

with open(text_file_path, 'r', encoding='utf-8') as text_file:
    new_questions = [line.strip() for line in text_file if line.strip()]

# Add new questions without duplicates
existing_questions_set = set(questions)
for question in new_questions:
    # 33% chance to remove the trailing question mark to improve variety
    if question.endswith('?') and random.random() < 0.33:
        question = question[:-1]
    if question not in existing_questions_set and question[:-1] not in existing_questions_set:
        questions.append(question)
        existing_questions_set.add(question)

# Update the data dictionary
data['questions'] = questions

# Write updated data back to the JSONC file
with open(jsonc_path, 'w', encoding='utf-8') as jsonc_file:
    commentjson.dump(data, jsonc_file, indent=4, ensure_ascii=False)

print(f"Successfully updated {jsonc_path} with new questions.")
