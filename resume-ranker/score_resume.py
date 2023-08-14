import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("ACCESS_TOKEN")

def score_job_description(job_title, job_description):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Rate this job description out of 10: {job_title} - {job_description}. Do not be lenient while rating and output only a single number out of 10 in first line and followed by explanation."
            }
        ],
        temperature=1,
        max_tokens=150,  # Increase max tokens to accommodate the explanation
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_content = response.choices[0].message['content']
    score_job_description = float(response_content.split('\n')[0])
    explanation = '\n'.join(response_content.split('\n')[1:])  # Extract the explanation

    data_jd = {
        "job_title": job_title,
        "old_job_description": job_description,
        "job_description_score": score_job_description,
        "explanation": explanation
    }

    return data_jd

# Load job title and description from a text file
with open("job_description.txt", "r") as file:
    lines = file.readlines()
    job_title = lines[0].strip().split(": ")[1]
    job_description = lines[1].strip().split(": ")[1]

# Read candidate information from the CSV file
candidates = []
with open("resumes_data.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        candidate_id = row["Candidate ID"]  # Replace with actual candidate ID column name
        candidate_data_jd = score_job_description(job_title, job_description)
        candidates.append((candidate_id, candidate_data_jd))

# Print the list of tuples containing candidate ID, data dictionary
for candidate_id, data in candidates:
    print(f"Candidate ID: {candidate_id}")
    print(data)

print("Resumes rated and scores stored in the list of tuples.")