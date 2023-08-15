from dotenv import load_dotenv
import os
import json
import openai

load_dotenv()
CANDIDATE_ANSWERS = os.getenv("CANDIDATE_ANSWERS")
SHORTLISTED_CV_QUESTIONS=os.getenv("SHORTLISTED_CV_QUESTIONS")
openai.api_key = os.getenv("ACCESS_TOKEN")

with open(CANDIDATE_ANSWERS, "r") as file:
    answer_data=json.load(file)

def score_candidate_answers(candidate_questions, candidate_answers):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"These are the 10 questions given to a candidate: {candidate_questions}. These are the answers provided by the candidate: {candidate_answers}. Rate these questions out of 10 and output only a number without providing an explanation. Average out all the scores."
            }
        ],
        temperature=1,
        max_tokens=20,  # Increase max tokens to accommodate the explanation
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response_content = response.choices[0].message['content']

    return response_content

def get_final_score():

    answers=answer_data[0]["answers"]
    candidate_id=answer_data[0]["candidate_id"]

    # candidate_id = request.form.get("candidate_id")

    with open(SHORTLISTED_CV_QUESTIONS, "r") as file:
        candidates_data = json.load(file)

    candidate_questions = None
    for a_candidate_details in candidates_data:
        all_candidate_ques_details = a_candidate_details[0]
        if all_candidate_ques_details["candidate_id"] == candidate_id:
            candidate_questions = all_candidate_ques_details["candidate_cv_questions"]
            break

    candidate_score=score_candidate_answers(candidate_questions,answers)

    return candidate_score

# print(get_final_score())