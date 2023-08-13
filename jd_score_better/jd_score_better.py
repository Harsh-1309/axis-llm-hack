import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("ACCESS_TOKEN")



job_title = "Data Scientist"
job_description = "We are seeking a skilled data scientist to join our team. You will be responsible for analyzing and interpreting complex data sets to discover actionable insights."


def scoreJobDescription(job_title, job_description):

    data_jd={}

    data_jd["job_title"]=job_title
    data_jd["old_job_description"]=job_description

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": f"Rate this job description out of 10: {job_description}. Do not be lenient while rating it and output only a number."
        }
    ],
    temperature=1,
    max_tokens=10,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
  
    score_job_description=response.choices[0].message.content
    data_jd["job_description_score"]=float(score_job_description)

    return data_jd

def improveJobDescription(job_title, job_description):

    data_jd={}

    data_jd["job_title"]=job_title
    data_jd["old_job_description"]=job_description

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[

        {
        "role": "user",
        "content": f"Make this job description better: {job_description}."
        }
    ],
    temperature=1,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    improved_job_description=response.choices[0].message.content
    data_jd["improved_job_description"]=improved_job_description

    return data_jd


# print(improve_job_description(job_title,job_description))