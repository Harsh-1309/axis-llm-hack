from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
from jd_score_better.jd_score_better import *
from resume_ranker.extract_resume_data import *
from resume_ranker.score_resume import *
from cv_questions.cv_questions import *
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey123!$#'
# openai.api_key = "YOUR_OPENAI_API_KEY"

from dotenv import load_dotenv

load_dotenv()
JD_SCORE_DATA_FILE = os.getenv("JD_SCORE_DATA_FILE")


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
        
        # return "Resumes extracted successfully." #data_
        flash("Resumes extracted successfully.")
        return redirect(url_for("score_resumes"))
    return render_template("extract_resumes.html")

@app.route("/score_resumes", methods=["GET", "POST"])
def score_resumes():
    if request.method == "POST":
        score_option = request.form.get("score_option")

        if score_option == "yes":
            data_=score_all_resumes()
            # return "Resumes scored successfully."
            flash("Resumes scored successfully.")
            return redirect(url_for("shortlist_candidates"))
        else:
            return "Resumes not scored."

    return render_template("score_resumes.html")

@app.route("/shortlist_candidates", methods=["GET", "POST"])
def shortlist_candidates():
    if request.method == "POST":
        num_candidates = int(request.form["num_candidates"])
        return redirect(url_for("generate_questions", num_candidates=int(num_candidates)))

    return render_template("shortlist_candidates.html")
    # return redirect(url_for("generate_questions", num_candidates=num_candidates))

@app.route("/generate_questions/<int:num_candidates>", methods=["GET", "POST"])
def generate_questions(num_candidates):
    if request.method == "POST":
        generate_option = request.form.get("generate_option")

        if generate_option == "yes":
            # Run your script to generate questions
            # Replace "generate_questions_script.py" with the actual script filename
            data_=generate_questions_for_shortlisted(num_candidates)
            return "Questions generated successfully."
        else:
            return "Questions not generated."

    return render_template("generate_questions.html", num_candidates=num_candidates)
if __name__ == "__main__":
    app.run(debug=True)