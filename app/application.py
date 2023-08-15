from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
from jd_score_better.jd_score_better import *
from resume_ranker.extract_resume_data import *
from resume_ranker.score_resume import *
from cv_questions.cv_questions import *
from cv_questions.rate_question_answers import *
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey123!$#'
from dotenv import load_dotenv

load_dotenv()
JD_SCORE_DATA_FILE = os.getenv("JD_SCORE_DATA_FILE")
SHORTLISTED_CV_QUESTIONS = os.getenv("SHORTLISTED_CV_QUESTIONS")
CANDIDATE_ANSWERS=os.getenv("CANDIDATE_ANSWERS")

def load_data():
    try:
        with open(JD_SCORE_DATA_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_data(data):
    with open(JD_SCORE_DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

@app.route("/", methods=["GET", "POST"])
def index():
    data = load_data()
    if request.method == "POST":

        job_title = request.form["job_title"]
        job_description = request.form["job_description"]
        accept_option = request.form.get("accept_option")
        score = scoreJobDescription(job_title,job_description)['job_description_score']
        improvement = improveJobDescription(job_title,job_description)['improved_job_description']

        if accept_option == "yes":
            accept_desc="yes"
        else:
            accept_desc="no"
        data=[]
        data.append(
            {
                "job_title": job_title,
                "old_description": job_description,
                "score": score,
                "new_description": improvement,
                "accept_new_desc": str(accept_desc)
            }
            
        )
        save_data(data)

        return render_template("index.html", job_title=job_title, job_description=job_description, score=score, improvement=improvement, accept_desc=accept_desc)
    return render_template("index.html")

@app.route("/extract_resumes", methods=["GET", "POST"])
def extract_resumes():
    if request.method == "POST":
        folder_path = request.form["folder_path"]
        data_=extract_all_resume_data(folder_path)

        flash("Resumes extracted successfully.")
        return redirect(url_for("score_resumes"))
    return render_template("extract_resumes.html")

@app.route("/score_resumes", methods=["GET", "POST"])
def score_resumes():
    if request.method == "POST":
        score_option = request.form.get("score_option")

        if score_option == "yes":
            data_=score_all_resumes()
            flash("Resumes scored successfully.")
            return redirect(url_for("shortlist_candidates"))
        else:
            return "Resumes not scored."

    return render_template("score_resumes.html")

@app.route("/shortlist_candidates", methods=["GET", "POST"])
def shortlist_candidates():
    if request.method == "POST":
        num_candidates = int(request.form["num_candidates"])
        return redirect(url_for("generate_questions", num_candidates=num_candidates))

    return render_template("shortlist_candidates.html")

@app.route("/generate_questions/<int:num_candidates>", methods=["GET", "POST"])
def generate_questions(num_candidates):
    if request.method == "POST":
        generate_option = request.form.get("generate_option")

        if generate_option == "yes":
            data_=generate_questions_for_shortlisted(num_candidates)
            return "Questions generated successfully."
        else:
            return "Questions not generated."

    return render_template("generate_questions.html", num_candidates=num_candidates)

@app.route("/answer_questions", methods=["GET", "POST"])
def answer_questions():
    if request.method == "POST":
        candidate_id = request.form.get("candidate_id")

        with open(SHORTLISTED_CV_QUESTIONS, "r") as file:
            candidates_data = json.load(file)

        candidate_questions = None
        for a_candidate_details in candidates_data:
            all_candidate_ques_details = a_candidate_details[0]
            if all_candidate_ques_details["candidate_id"] == candidate_id:
                candidate_questions = all_candidate_ques_details["candidate_cv_questions"]
                break

        if candidate_questions:
            if request.method == "POST":
                answers = {question_id: request.form.get(f"answer_{question_id}") for question_id in candidate_questions.keys()}
                with open(CANDIDATE_ANSWERS, "r") as file:
                    candidates_answers_data = json.load(file)

                data_to_append = {"candidate_id": candidate_id, "answers": answers}
                candidates_answers_data.append(data_to_append)

                # with open(CANDIDATE_ANSWERS, "w") as file:
                #     json.dump(candidates_answers_data, file, indent=4)

                flash("Answers submitted successfully.", "success")
                return render_template("input_candidate_answers.html", candidate_id=candidate_id, questions=candidate_questions)
                # return redirect(url_for("thank_you", candidate_id=candidate_id, score=score))

        else:
            flash("Candidate ID not found.", "error")
            return redirect(url_for("answer_questions"))

    return render_template("input_candidate_id.html")

@app.route("/display_score")
def display_score():
    candidate_score=get_final_score()
    return render_template("display_score.html", score=candidate_score)

if __name__ == "__main__":
    app.run(debug=True)