import os
from dotenv import load_dotenv
import openai


load_dotenv()
openai.api_key = os.getenv("ACCESS_TOKEN")



# job_title = "Data Scientist"
# job_description = "We are seeking a skilled data scientist to join our team. You will be responsible for analyzing and interpreting complex data sets to discover actionable insights."
# cv="Jyotir Aditya   Bangalore|  LinkedIn  | +91 8439622759 | adityajyotir05@gmail.com   EDUCATION   Vellore Institute of Technology  Vellore, India   Bachelor of Technology - Computer Science and Engineering; GPA: 8.35  July 2019 - June 2023   ●  Courses: OS, DSA, Data Visualisation, Artificial Intelligence, Machine Learning, Networking, Databases,   Internet and Web Programming, Business Mathematics   SKILLS SUMMARY   ●  Languages  : Python, C++, JavaScript, SQL, Bash, JAVA,  Solidity   ●  Frameworks:  Scikit, NLTK, SpaCy, TensorFlow, Keras,  Django, Flask, ReactJS, NodeJS, MongoDB, OpenCV   ●  Tools:  PowerBI, GIT, MySQL, Collab, Postman, Adobe  Suite, FIGMA   ●  Platforms:  Linux, Web, Windows, AWS, GCP   ●  Soft Skills:  Leadership, Event Management, Writing,  Public Speaking, Time Management   EXPERIENCE   Ericsson Global India  Bangalore   Graduate Engineer Trainee  Feb 2023-Current   ●  Utilized  Python,  Pandas,  NumPy,  and  TensorFlow  Extended  (TFX)  to  develop  and  implement  data   preprocessing pipelines for efficient data handling and transformation.   ●  Applied  Azure  services  and  Docker  to  deploy  and  manage  data  processing  and  machine  learning  models  in  a   cloud environment.   ●  Worked  on  data  preprocessing  tasks,  including  data  cleaning,  feature  engineering,  and  data  normalization,  to   ensure high data quality and reliability for downstream analysis and modeling.   ●  Collaborated  with  cross-functional  teams  to  design  and  implement  Machine  Learning  Operations  (MLOps)   practices, including model versioning, automated model deployment, and monitoring.   IMS Healthcare (IQVIA)  Bangalore   Full Term Intern  Aug’22-Jan’23   ●  Proficiently  utilized  Python,  Django,  Pandas,  and  NumPy  to  develop  and  maintain  robust  and  scalable  web   applications.   ●  Implemented  unit  test  cases  to  ensure  the  quality  and  functionality  of  the  developed  codebase,  promoting  a   reliable and bug-free software environment.   ●  Collaborated  with  the  development  team  to  design  and  develop  REST  APIs,  providing  seamless  integration   between different components of the application.   ●  Integrated  machine  learning  models  into  the  existing  system,  enabling  predictive  analytics  and  enhancing  the   overall functionality of the software.   PROJECTS   ●  A  unique  approach  to  Malware  Analysis  using  Deep  Learning  (Presenting  research  paper  at  IEEE   ICECIE’2022):  Research  done  was  based  on  optimizing  Malware  detection  by  creating  a  grayscale  image  of   52  attributes  received  through  static  analysis  of  the  malware  file  and  applying  different  models  over  it.  Impact:   Significant improve in the accuracy over previous works   ●  YogAI  -  AI  based  Yoga  Platform  (MLP,  openCV,  HRNet,  Image  Processing):  Made  an  AI  platform  to  help   customers  perform  yoga  at  the  ease  of  their  home.  Takes  live  video  as  input  and  outputs  the  Accuracy  of  the   Aasan performed. (Tech: Python,OvenCV, Flask)(December 2021)   ●  Databreach  Visualization  (Python,  HTML,  Ploty,  Jinja,  Flask,  Numpy,  Pandas):  Ituative  Platform  to  learn   about  recent  trends  of  Data  breaches  in  the  world  with  Ploty  visualizations  to  make  interactive  and   helpful.(April 2021)   HONORS   ●  Identified  as  a  Special  Achiever  for  the  21-22  by  VIT,Vellore  •  Awarded  as  a  full  time  Internship  in  Kestone   Global for my graphic designing skills.   ●  Volunteered  as  a  member  for  FEPSI  Vellore,  India:  Helped  underprivileged  children  learn  subjects  like  Math  and   helped deciding a career path"

def generateInterviewQuestions(job_title, job_description,cv):

    data_jd={}

    data_jd["job_title"]=job_title
    data_jd["job_description"]=job_description

    if len(cv)>=4096:
        cv=cv[:3400]

    data_jd["resume"]=cv

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": f"This is a resume of a candidate: {cv}. He/she is applying for this job title: {job_title} with this job description: {job_description}. Generate appropriate 10 questions which can be asked in first round of interview."
        }
    ],
    temperature=1,
    max_tokens=400,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
  
    generated_questions=response.choices[0].message.content
    data_jd["job_interview_generated_questions"]=generated_questions

    return data_jd


print(generateInterviewQuestions(job_title,job_description,cv))