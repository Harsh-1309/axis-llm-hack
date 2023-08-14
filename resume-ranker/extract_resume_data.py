import os
import csv
import PyPDF2
import re  # Import the regex module
from docx import Document
import re
import zipfile
import uuid
import json
from dotenv import load_dotenv

load_dotenv()
CANDIDATE_RESUME_DATA_FILE = os.getenv("CANDIDATE_RESUME_DATA_FILE")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def remove_empty_lines(text):
    return text.replace("\n", " ")

def extract_emails(text):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    emails = re.findall(email_pattern, text)
    return emails

def process_resumes(folder_path):
    resumes_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        candidate_id = str(uuid.uuid4())
        
        if filename.lower().endswith(".pdf"):
            candidate_name = os.path.splitext(filename)[0]  # Extract candidate name without extension
            resume_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith(".docx"):
            candidate_name = os.path.splitext(filename)[0]
            resume_text = extract_text_from_docx(file_path)
        elif filename.lower().endswith(".txt"):
            candidate_name = os.path.splitext(filename)[0]
            with open(file_path, "r") as txt_file:
                resume_text = txt_file.read()
        elif filename.lower().endswith(".zip"):
            candidate_name = os.path.splitext(filename)[0]
            zip_folder = os.path.join(folder_path, candidate_name)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(zip_folder)
            resume_texts = []
            for nested_filename in os.listdir(zip_folder):
                nested_file_path = os.path.join(zip_folder, nested_filename)
                if nested_filename.lower().endswith((".pdf", ".docx", ".txt")):
                    nested_resume_text = ""
                    if nested_filename.lower().endswith(".pdf"):
                        nested_resume_text = extract_text_from_pdf(nested_file_path)
                    elif nested_filename.lower().endswith(".docx"):
                        nested_resume_text = extract_text_from_docx(nested_file_path)
                    elif nested_filename.lower().endswith(".txt"):
                        with open(nested_file_path, "r") as txt_file:
                            nested_resume_text = txt_file.read()
                    resume_texts.append(nested_resume_text)
            combined_resume_text = "\n".join(resume_texts)
            cleaned_resume_text = remove_empty_lines(combined_resume_text)
            emails = extract_emails(cleaned_resume_text)
            resumes_data.append((candidate_id, candidate_name, cleaned_resume_text, ", ".join(emails)))
            # Clean up extracted zip folder
            os.remove(file_path)
            os.rmdir(zip_folder)
        else:
            print(f"Unsupported format for file: {filename}")
            continue
        
        cleaned_resume_text = remove_empty_lines(resume_text)
        emails = extract_emails(cleaned_resume_text)
        resumes_data.append((candidate_id,candidate_name, cleaned_resume_text, ", ".join(emails)))

    
    return resumes_data

def load_data():
    try:
        with open(CANDIDATE_RESUME_DATA_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_data(data):
    with open(CANDIDATE_RESUME_DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

def extract_all_resume_data(folder_path="all_candidate_resumes/"):
    resumes_data = process_resumes(folder_path)
    data = load_data()

    for a_candidate in resumes_data:
        data.append([
                {
                    "candidate_id": a_candidate[0],
                    "candidate_name": a_candidate[1],
                    "candidate_email": a_candidate[3],
                    "candidate_cv": a_candidate[2]
                }
                ]
            )
    save_data(data)

    return "All extraction successful", data

# print(extract_all_resume_data())
