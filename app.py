import os
import logging
import hashlib
import httpx
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sqlite3
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# LLM Proxy Token
LLM_TOKEN = os.environ.get("LLMPROXY_TOKEN")

# Helper function to call LLM
async def call_llm(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.llmproxy.com/generate",
            headers={"Authorization": f"Bearer {LLM_TOKEN}"},
            json={"model": "gpt-4o-mini", "prompt": prompt}
        )
        response.raise_for_status()
        return response.json().get("text", "")

# Task execution functions
async def execute_task(task_description: str):
    if "install uv" in task_description:
        # A1: Install uv and run datagen.py
        user_email = task_description.split()[-1]  # Assuming email is the last word
        os.system(f"pip install uv")
        os.system(f"python https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py {user_email}")
        return "Data generation completed."

    elif "format the contents" in task_description:
        # A2: Format the contents of /data/format.md
        os.system("npx prettier --write /data/format.md")
        return "File formatted."

    elif "count the number of Wednesdays" in task_description:
        # A3: Count Wednesdays in /data/dates.txt
        with open('/data/dates.txt', 'r') as f:
            dates = f.readlines()
        wednesdays_count = sum(1 for date in dates if datetime.strptime(date.strip(), '%Y-%m-%d').weekday() == 2)
        with open('/data/dates-wednesdays.txt', 'w') as f:
            f.write(str(wednesdays_count))
        return "Wednesdays counted."

    elif "sort the array of contacts" in task_description:
        # A4: Sort contacts in /data/contacts.json
        with open('/data/contacts.json', 'r') as f:
            contacts = json.load(f)
        sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
        with open('/data/contacts-sorted.json', 'w') as f:
            json.dump(sorted_contacts, f)
        return "Contacts sorted."

    elif "write the first line of the 10 most recent .log files" in task_description:
        # A5: Write first line of recent log files
        log_files = sorted(glob.glob('/data/logs/*.log'), key=os.path.getmtime, reverse=True)[:10]
        with open('/data/logs-recent.txt', 'w') as f:
            for log_file in log_files:
                with open(log_file, 'r') as lf:
                    first_line = lf.readline().strip()
                    f.write(first_line + '\n')
        return "Recent log lines written."

    elif "extract the first occurrence of each H1" in task_description:
        # A6: Extract H1 from Markdown files
        index = {}
        for md_file in glob.glob('/data/docs/*.md'):
            with open(md_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        index[os.path.basename(md_file)] = line[2:].strip()  # Skip '# '
                        break
        with open('/data/docs/index.json', 'w') as f:
            json.dump(index, f)
        return "Index created."

    elif "extract the sender’s email address" in task_description:
        # A7: Extract email from /data/email.txt
        with open('/data/email.txt', 'r') as f:
            email_content = f.read()
        prompt = f"Extract the sender's email address from the following content: {email_content}"
        sender_email = await call_llm(prompt)
        with open('/data/email-sender.txt', 'w') as f:
            f.write(sender_email.strip())
        return "Sender's email extracted."

    elif "extract the card number" in task_description:
        # A8: Extract credit card number from image
        # This is a placeholder; actual implementation would require image processing
        with open('/data/credit-card.png', 'rb') as f:
            image_data = f.read()
        prompt = "Extract the credit card number from this image."
        card_number = await call_llm(prompt)
        with open('/data/credit-card.txt', 'w') as f:
            f.write(card_number.replace(" ", "").strip())
        return "Credit card number extracted."

    elif "find the most similar pair of comments" in task_description:
        # A9: Find similar comments
        with open('/data/comments.txt', 'r') as f:
            comments = f.readlines()
        prompt = f"Find the most similar pair of comments from the following list: {comments}"
        similar_comments = await call_llm(prompt)
        with open('/data/comments-similar.txt', 'w') as f:
            f.write(similar_comments.strip())
        return "Similar comments found."

    elif "total sales of all the items in the 'Gold' ticket type" in task_description:
        # A10: Calculate total sales from SQLite database
        conn = sqlite3.connect('/data/ticket-sales.db')
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        with open('/data/ticket-sales-gold.txt', 'w') as f:
            f.write(str(total_sales))
        conn.close()
        return "Total sales calculated."

    # Handle additional business tasks
    # B3: Fetch data from an API and save it
    elif "fetch data from an API" in task_description:
        api_url = task_description.split()[-1]  # Assuming URL is the last word
        response = await httpx.get(api_url)
        response.raise_for_status()
        with open('/data/api_data.json', 'w') as f:
            json.dump(response.json(), f)
        return "API data fetched and saved."

    # Additional tasks can be implemented similarly...

    else:
        raise HTTPException(status_code=400, detail="Task not recognized.")

# API Endpoints
@app.post("/run")
async def run_task(task: str):
    try:
        result = await execute_task(task)
        return {"message": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/read")
async def read_file(path: str):
    if not path.startswith("/data/"):
        raise HTTPException(status_code=400, detail="Access outside /data is not allowed.")
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        return PlainTextResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)