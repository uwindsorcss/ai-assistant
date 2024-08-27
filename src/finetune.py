"""
    Pipeline for fine-tuning GPT-4o-mini on our custom dataset
    ONLY RUN IF YOU ARE RETRAINING THE LLM.
    If regenerating the dataset ensure questions.jsonc is prepared and add: --generate
    To output the unranked documents for each question add: --documents
    If working with a file not yet uploaded to openai put the training.jsonl file into ./data/finetune/ and add: --upload
    If retraining on an existing file uploaded to OpenAI add: --file=<file_id>
    If retraining the model add: --finetune
"""
import sys
import json
import commentjson
from uwin_ai_assistant import config
from uwin_ai_assistant.clients import openai_client
from uwin_ai_assistant.inference import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, generate_response

QUESTIONS_PATH = "./data/finetune/questions.jsonc" # Path to base questions file
TRAINING_PATH = "./data/finetune/training.jsonl" # Path to generated training dataset
DOCUMENTS_PATH = "./data/finetune/documents.json" # Path to save retrieved documents

cli_args = sys.argv[1:]

# Generate dataset. This takes in a questions.jsonc file and generates a training.jsonl file. Update responses in training.jsonl as needed.
if "--generate" in cli_args:
    # Format of each training example
    format = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    }

    # Load questions list
    saved_documents = {}
    with open(QUESTIONS_PATH, "r") as questions_file, open(TRAINING_PATH, "a") as training_file:
        questions = commentjson.load(questions_file)["questions"]
        for question in questions:
            response, documents, unranked_documents = generate_response(question, return_documents=True, return_unranked_documents=True) # Get response and documents
            prompt = USER_PROMPT_TEMPLATE % (documents, question) # Format prompt
            if "--documents" in cli_args:
                saved_documents[question] = unranked_documents
            format["messages"][1]["content"] = prompt # Insert query, documents into training example
            format["messages"][2]["content"] = response # Insert response into training example
            json.dump(format, training_file, indent=4) # Save training example

    if "--documents" in cli_args:
        with open(DOCUMENTS_PATH, "w") as documents_file:
            json.dump(saved_documents, documents_file, indent=4)

    print("Successfully generated training dataset. Tweak responses as needed for the finetune.")

# Upload dataset to OpenAI
if "--upload" in cli_args:
    file_response = openai_client.files.create(
        file=open("./data/finetune/training.jsonl", "rb"),
        purpose="fine-tune",
    )
    file_id = file_response["id"]
    print(f"Successfully uploaded training dataset to OpenAI with file id {file_id}.")
elif "--finetune" in cli_args:
    file_id = None
    for arg in cli_args:
        if arg.startswith("--file="):
            file_id = arg.split("=")[1]
            break
    if file_id == None:
        print("Please provide a file id for the finetune with --file=<file_id>")
        sys.exit(1)
    print(f"Using file id {file_id} for the finetune.")

# Submit finetuning job
if "--finetune" in cli_args:
    finetune_response = openai_client.fine_tuning.jobs.create(
        training_file=file_id,
        model=config.GPT_MODEL,
        suffix="uwindsor_chatbot",
    )
    job_id = finetune_response["id"]
    print(f"Successfully ran finetuning job with job id {job_id}.")
