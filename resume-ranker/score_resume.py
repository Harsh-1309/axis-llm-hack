import openai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
RANKED_RESUMES = os.getenv("RANKED_RESUMES")
CANDIDATE_RESUME_DATA_FILE=os.getenv("CANDIDATE_RESUME_DATA_FILE")
JD_SCORE_DATA_FILE=os.getenv("JD_SCORE_DATA_FILE")
openai.api_key = os.getenv("ACCESS_TOKEN")


# print(data)

def score_resume(job_title, job_description, cv):
    if len(cv)>=4096:
        cv=cv[:3400]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Rate this resume out of 10: {cv}. The job title and description for this are: {job_title} and {job_description} respectively. Output only a single number out of 10 in first line and followed by explanation."
            }
        ],
        temperature=1,
        max_tokens=200,  # Increase max tokens to accommodate the explanation
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_content = response.choices[0].message['content']
    resume_score = response_content.split('\n')[0]
    numbers = re.findall(r'\d+\.\d+|\d+', resume_score)  # Find all sequences of digits
    if numbers:
        resume_score = float(numbers[0])  # Convert the first found number to an integer
    explanation = '\n'.join(response_content.split('\n')[1:])  # Extract the explanation

    data_resume = {
        "job_title": job_title,
        "job_description": job_description,
        "resume_score": resume_score,
        "explanation": explanation
    }

    return data_resume

def score_all_resumes():
    with open(CANDIDATE_RESUME_DATA_FILE, "r") as file:
        candidate_resume_data=json.load(file)

    with open(RANKED_RESUMES, "r") as file:
        ranked_resumes=json.load(file)

    with open(JD_SCORE_DATA_FILE, "r") as file:
        jd_data=json.load(file)

    job_title=jd_data[0]["job_title"]

    if jd_data[0]["accept_new_desc"]=="yes":
        job_description=jd_data[0]["new_description"]
    else:
        job_description=jd_data[0]["old_description"]

    for a_candidate in candidate_resume_data:
        for details in a_candidate:
            candidate_id=details["candidate_id"]
            candidate_cv=details["candidate_cv"]

            resume_score_explanation_data=score_resume(job_title,job_description,candidate_cv)
            data_to_append=[
                {
                    "candidate_id": candidate_id,
                    "candidate_cv": candidate_cv,
                    "candidate_cv_score": resume_score_explanation_data["resume_score"],
                    "candidate_score_explanation": resume_score_explanation_data["explanation"]
                }
                ]

            ranked_resumes.append(data_to_append)

            with open(RANKED_RESUMES, "w") as file:
                json.dump(ranked_resumes, file, indent=4)

            return data_to_append



print(score_all_resumes())
























# # Load job title and description from a text file
# with open("job_description.txt", "r") as file:
#     lines = file.readlines()
#     job_title = lines[0].strip().split(": ")[1]
#     job_description = lines[1].strip().split(": ")[1]

# # Read candidate information from the CSV file
# candidates = []
# with open("resumes_data.csv", "r", newline="", encoding="utf-8") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         candidate_id = row["Candidate ID"]  # Replace with actual candidate ID column name
#         candidate_data_jd = score_job_description(job_title, job_description)
#         candidates.append((candidate_id, candidate_data_jd))

# # Print the list of tuples containing candidate ID, data dictionary
# for candidate_id, data in candidates:
#     print(f"Candidate ID: {candidate_id}")
#     print(data)

# print("Resumes rated and scores stored in the list of tuples.")