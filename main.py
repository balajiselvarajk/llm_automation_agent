import os
import json
import re
import base64
import sqlite3
import subprocess
from datetime import datetime
from urllib import response
from dotenv import load_dotenv

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load API token from environment variable
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImJhbGFqaS5zZWx2YXJhakBzdHJhaXZlLmNvbSJ9.k_dAv1Dk5HkY_COXzjDDne-2Z5FmbpF7RKkH7ECO51Q"

# Load environment variables from .env file
load_dotenv()

# Access the AIPROXY_TOKEN variable
AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# A1: Install uv (if required) and run the datagen script with user_email as the argument.
def install_uv_and_run_datagen(script_url, email):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    command = f"uv run {script_url} {email}"
    subprocess.run(command, shell=True)

# A2: Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place.
def format_contents_with_prettier(file_path):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    subprocess.run("npm install prettier@3.4.2", shell=True)
    subprocess.run(f"npx prettier@3.4.2 --write {file_path}", shell=True)

# Function to parse a string to extract a date
def parse_date_string(date_str):
    for dateformat in ("%b %d, %Y", "%d-%b-%Y", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, dateformat)
        except ValueError:
            continue
    raise ValueError(f"time data '{date_str}' does not match any known format")

# A3: Count the number of Wednesdays in /data/dates.txt and write the number to /data/dates-wednesdays.txt
def count_wednesdays_in_dates(file_path, output_folder):
    with open(file_path, "r") as f:
        dates = f.readlines()
    wednesdays = sum(1 for date in dates if parse_date_string(date.strip()).weekday() == 2)
    with open(output_folder, "w") as f:
        f.write(str(wednesdays))

# A4: Sort the array of contacts in /data/contacts.json by last_name, then first_name
def sort_contacts_by_name(input_path, output_folder):
    with open(input_path, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
    with open(output_folder, "w") as f:
        json.dump(sorted_contacts, f, indent=4)

# A5: Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt
def write_recent_log_first_lines(input_folder, output_folder):
    log_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".log")],
        key=lambda x: os.path.getmtime(os.path.join(input_folder, x)),
        reverse=True
    )
    with open(output_folder, "w") as f:
        for log_file in log_files[:10]:
            with open(os.path.join(input_folder, log_file), "r") as lf:
                f.write(lf.readline())

# A6: Create an index file mapping filenames to their titles from Markdown (.md) files
def create_markdown_index(input_folder, output_folder):
    index = {}
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file] = line[2:].strip()
                            break
    with open(output_folder, "w") as f:
        json.dump(index, f, indent=4)

# A7: Extract the sender’s email address from /data/email.txt
def extract_email_sender(input_path, output_folder):
    with open(input_path, "r") as f:
        email_content = f.read()
    
    prompt = f"Extract the sender's email address from the following email content:\n\n{email_content}"
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the sender's email address from the following email content"},
            {"role": "user", "content": email_content}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sendersEmailAdress": {"type": "string"}
                    },
                    "required": ["sendersEmailAdress"],
                    "additionalProperties": False
                }
            }
        }
    }

    response = requests.post(url=url, headers=headers, json=data, verify=False)
    response_json = response.json()
    email_address_dict = json.loads(response_json['choices'][0]['message']['content'].strip())
    email_address = email_address_dict['sendersEmailAdress']
    
    with open(output_folder, "w") as f:
        f.write(email_address)

# A8: Extract the credit card number from /data/credit-card.png
def extract_credit_card_number(input_path, output_folder):
    # Read the image file and encode it to base64
    with open(input_path, 'rb') as f:
        binary_data = f.read()
        image_b64 = base64.b64encode(binary_data).decode()

    # Create a data URI for the image
    data_uri = f"data:image/png;base64,{image_b64}"
    
    # Define the API endpoint and headers
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    # Prepare the data payload for the API request
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the number from the image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ]
    }

    # Make the API request
    response = requests.post(url=url, headers=headers, json=data, verify=False)

    # Check for a successful response
    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        pattern = r'\b(?:\d{4}[\s-]?){3}\d{4}\b'
        match = re.search(pattern, content)
        if match:
            credit_card_number = match.group(0).replace(" ", "").replace("-", "")  # Remove spaces and dashes
            print("Extracted Credit Card Number:", credit_card_number)
            
            # Write the credit card number to the output file without spaces
            with open(output_folder, 'w') as output_file:
                output_file.write(credit_card_number)
        else:
            print("No credit card number found.")        
    else:
        print(f"Error: {response.status_code} - {response.text}")

# A9: Find the most similar pair of comments in /data/comments.txt
def find_most_similar_comments(input_location: str, output_location: str):
    with open(input_location, "r") as f:
        comments = [i.strip() for i in f.readlines()]
    
    model = SentenceTransformer("all-MiniLM-L12-v2")
    embeddings = model.encode(comments)
    similarity_mat = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_mat, 0)
    max_index = int(np.argmax(similarity_mat))
    i, j = max_index // len(comments), max_index % len(comments)
    
    with open(output_location, "w") as g:
        g.write(comments[i] + "\n" + comments[j])
    
    return {"status": "Successfully Created", "output_file destination": output_location}

# A10: Calculate the total sales of all items in the "Gold" ticket type
def calculate_gold_ticket_sales(input_path, output_folder):
    conn = sqlite3.connect(input_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = 'gold'")
    total_sales = cursor.fetchone()[0]
    
    with open(output_folder, "w") as f:
        f.write(str(total_sales))
    
    conn.close()

# Define the task prompt descriptions
json_task_prompt_desc = [
    {
        "type": "function",
        "function": {
            "name": "install_uv_and_run_datagen",
            "description": "Install 'uv' (if required) and run the datagen script with 'user_email' as the argument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the datagen script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "User email as arguments to pass to the script."
                    }
                },
                "required": ["script_url", "args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_contents_with_prettier",
            "description": "Format the contents of '/data/format.md' using 'prettier@3.4.2', updating the file in place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The file path of the markdown file to format."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_wednesdays_in_dates",
            "description": "Count the number of Wednesdays in '/data/dates.txt' and write the number to '/data/dates-wednesdays.txt'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Open and read the contents of '/data/dates.txt'."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "Write the count of Wednesdays to '/data/dates-wednesdays.txt'."
                    }
                },
                "required": ["file_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts_by_name",
            "description": "Sort the array of contacts in '/data/contacts.json' by last name, then first name, and write the result to '/data/contacts-sorted.json'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing contacts."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the sorted contacts."
                    }
                },
                "required": ["input_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_log_first_lines",
            "description": "Write the first line of the 10 most recent .log files in '/data/logs/' to '/data/logs-recent.txt', most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_folder": {
                        "type": "string",
                        "description": "The directory that contains the log files."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file where the extracted lines will be written."
                    }
                },
                "required": ["input_folder", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_markdown_index",
            "description": "Find all Markdown (.md) files in '/data/docs/' and create an index file '/data/docs/index.json' mapping filenames to their titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_folder": {
                        "type": "string",
                        "description": "The directory containing Markdown files."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the index."
                    }
                },
                "required": ["input_folder", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_sender",
            "description": "Extract the sender’s email address from '/data/email.txt' and write it to '/data/email-sender.txt'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing the email."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the extracted email address."
                    }
                },
                "required": ["input_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract the credit card number from '/data/credit_card.png' and write it without spaces to '/data/credit-card.txt'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the image file containing the credit card number."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the extracted credit card number."
                    }
                },
                "required": ["input_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_most_similar_comments",
            "description": "Find the most similar pair of comments in '/data/comments.txt' and write them to '/data/comments-similar.txt'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the file containing comments."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the similar comments."
                    }
                },
                "required": ["input_path", "output_folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_gold_ticket_sales",
            "description": "Calculate the total sales of all items in the 'Gold' ticket type from '/data/ticket-sales.db' and write the number to '/data/ticket-sales-gold.txt'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "The path of the SQLite database file."
                    },
                    "output_folder": {
                        "type": "string",
                        "description": "The path of the file to write the total sales."
                    }
                },
                "required": ["input_path", "output_folder"]
            }
        }
    }
]

@app.get("/")
def home():
    return {"message": "Project 1 - LLM-based Automation Agent!"}

@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/run")
def task_runner(task: str):
    FORBIDDEN_PATTERNS = [
        r"\bdelete\b", r"\bremove\b", r"\berase\b", r"\bdestroy\b", r"\bdrop\b", r"\brm -rf\b",
        r"\bunlink\b", r"\brmdir\b", r"\bdel\b", 
        r"\bos\s*\.\s*remove\b", 
        r"\bos\s*\.\s*unlink\b", 
        r"\bshutil\s*\.\s*rmtree\b", 
        r"\bpathlib\s*\.\s*Path\s*\.\s*unlink\b"
    ]

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, task, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Task contains forbidden operations (deletion is not allowed).")
    
    url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": task},
            {
                "role": "system",
                "content": """
You are an assistant who has to perform various tasks based on the user's request.
You have access to the following tools:
1. install_uv_and_run_datagen: Install uv (if required) and run the datagen script with user_email as the argument.
2. format_contents_with_prettier: Format a file using prettier.
3. count_wednesdays_in_dates: Count the number of Wednesdays in /data/dates.txt and write the number to /data/dates-wednesdays.txt.
4. sort_contacts_by_name: Sort the array of contacts in /data/contacts.json by last_name, then first name, and write the result to /data/contacts-sorted.json.
5. write_recent_log_first_lines: Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.
6. create_markdown_index: Find all Markdown (.md) files in /data/docs/ and create an index file /data/docs/index.json mapping filenames to their titles.
7. extract_email_sender: Extract the sender’s email address from /data/email.txt and write it to /data/email-sender.txt.
8. extract_credit_card_number: Extract the credit card number from /data/credit_card.png and write it without spaces to /data/credit-card.txt.
9. find_most_similar_comments: Find the most similar pair of comments in /data/comments.txt and write them to /data/comments-similar.txt.
10. calculate_gold_ticket_sales: Calculate the total sales of all items in the "Gold" ticket type from /data/ticket-sales.db and write the number to /data/ticket-sales-gold.txt.

Use the appropriate tool based on the task description provided by the user.
                """
            }
        ],
        "tools": json_task_prompt_desc,
        "tool_choice": "auto"
    }

    response = requests.post(url=url, headers=headers, json=data)
    response_json = response.json()
    print(response_json)
    execute_task_call_info = response_json['choices'][0]['message']['tool_calls'][0]['function']

    execute_task_by_name = execute_task_call_info['name']
    arguments = json.loads(execute_task_call_info['arguments'])
    path_keys = ["input_path", "output_folder", "input_folder", "file_path"]

    for key in path_keys:
        if key in arguments:
            if not arguments[key].startswith("/data"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid path '{arguments[key]}': Data outside /data is never accessed or exfiltrated."
                )

    task_mapping = {
        "install_uv_and_run_datagen": install_uv_and_run_datagen,
        "format_contents_with_prettier": format_contents_with_prettier,
        "count_wednesdays_in_dates": count_wednesdays_in_dates,
        "sort_contacts_by_name": sort_contacts_by_name,
        "write_recent_log_first_lines": write_recent_log_first_lines,
        "create_markdown_index": create_markdown_index,
        "extract_email_sender": extract_email_sender,
        "extract_credit_card_number": extract_credit_card_number,
        "find_most_similar_comments": find_most_similar_comments,
        "calculate_gold_ticket_sales": calculate_gold_ticket_sales,
    }

    if execute_task_by_name in task_mapping:
        task_function = task_mapping[execute_task_by_name]
        if execute_task_by_name == "install_uv_and_run_datagen":
            task_function(arguments['script_url'], arguments['args'][0])
        else:
            task_function(arguments['input_path'], arguments['output_folder'])
    else:
        raise HTTPException(status_code=400, detail="Invalid task name.")

    return execute_task_call_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)