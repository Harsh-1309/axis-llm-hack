import re
import os
from dotenv import load_dotenv
import json

load_dotenv()
RANKED_RESUMES = os.getenv("RANKED_RESUMES")
CANDIDATE_RESUME_DATA_FILE=os.getenv("CANDIDATE_RESUME_DATA_FILE")

def get_candidate_resume(candidate_id):

    with open(CANDIDATE_RESUME_DATA_FILE, "r") as file:
        candidate_resume_data=json.load(file)

    data=candidate_resume_data
    candidates = [candidate for sublist in data for candidate in sublist]
    candidate = next((c for c in candidates if c["candidate_id"] == candidate_id), None)
    
    if candidate:
        return candidate["candidate_cv"]
    else:
        return "Candidate not found."

print(get_candidate_resume("9dc84545-4eb3-43b9-80d4-2d858dfecc63"))
