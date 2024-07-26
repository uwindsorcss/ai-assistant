"""
    Pipeline for fine-tuning GPT-4o-mini on our custom dataset
    ONLY RUN IF YOU ARE RETRAINING THE LLM.
    If regenerating the dataset ensure questions.jsonc is prepared and add: --generate
    If working with a file not yet uploaded to openai put the training.jsonl file into ./data/finetune/ and add: --upload
    If retraining on an existing file uploaded to OpenAI add: --file=<file_id>
    If retraining the model add: --finetune
"""
import sys
import json
from uwin_ai_assistant.clients import openai_client
from uwin_ai_assistant.inference import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, generate_response

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
    with open("./data/finetune/questions.jsonc", "r") as questions_file, open("./data/finetune/training.jsonl", "a") as training_file:
        questions = json.load(questions_file)["questions"]
        for question in questions:
            response, documents = generate_response(question, return_documents=True) # Get response and documents
            prompt = USER_PROMPT_TEMPLATE % (documents, question) # Format prompt
            format["messages"][1]["content"] = prompt # Insert query, documents into training example
            format["messages"][2]["content"] = response # Insert response into training example
            training_file.write(json.dumps(format) + "\n") # Save training example

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
        model="gpt-4o-mini",
        suffix="uwindsor_chatbot",
    )
    job_id = finetune_response["id"]
    print(f"Successfully ran finetuning job with job id {job_id}.")
