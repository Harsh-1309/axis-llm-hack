from flask import Flask, render_template, request, jsonify
import json
from jd_score_better.jd_score_better import *
import openai

app = Flask(__name__)
# openai.api_key = "YOUR_OPENAI_API_KEY"

JD_SCORE_DATA_FILE = "app/all_resume_data/jd_score_imp.json"

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
        accept_desc=request.form["accept_desc"]
        score = scoreJobDescription(job_title,job_description)['job_description_score']
        improvement = improveJobDescription(job_title,job_description)['improved_job_description']
        
        accepted_values=["yes","Yes","y","Y"]
        if accept_desc in accepted_values:
            accept_desc=True
        else:
            accept_desc=False

        data.append([
            {
                "job_title": job_title,
                "old_description": job_description,
                "score": score,
                "new_description": improvement,
                "accept_new_desc": str(accept_desc)
            }
            ]
        )
        save_data(data)

        return render_template("index.html", job_title=job_title, job_description=job_description, score=score, improvement=improvement)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
