from dotenv import load_dotenv
import os
import json

load_dotenv()
RANKED_RESUMES = os.getenv("RANKED_RESUMES")

def load_data():
    try:
        with open(RANKED_RESUMES, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def get_candidate_ids_top_n(n):

    data=load_data()
    candidates = [candidate for sublist in data for candidate in sublist]
    sorted_candidates = sorted(candidates, key=lambda x: x["candidate_cv_score"], reverse=True)
    
    if n<=len(data):
        top_n_candidate_ids = [candidate["candidate_id"] for candidate in sorted_candidates[:n]]
    else:
        top_n_candidate_ids = [candidate["candidate_id"] for candidate in sorted_candidates[:]]

    return top_n_candidate_ids

# Number of top candidates to retrieve
# top_n = 5

# top_candidate_ids = get_candidate_ids_top_n(top_n)
# print("Top", top_n, "candidate IDs:", top_candidate_ids)
