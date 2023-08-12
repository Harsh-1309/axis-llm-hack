import spacy
import fitz

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        num_pages = pdf_document.page_count
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def generate_questions(resume_text):
    doc = nlp(resume_text)
    
    questions = []
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            questions.append(f"What are {ent.text}'s skills?")
        elif ent.label_ == "ORG":
            questions.append(f"What roles did {ent.text} have in the past?")
        elif ent.label_ == "DATE":
            questions.append(f"When did {ent.text} work at previous positions?")
        elif ent.label_ == "GPE":
            questions.append(f"What projects has {ent.text} worked on?")
    
    return questions

def main():
    pdf_path = "path_to_your_resume.pdf"
    resume_text = extract_text_from_pdf(pdf_path)
    
    questions = generate_questions(resume_text)
    
    print("Extracted Resume Text:")
    print(resume_text)
    print("\nGenerated Questions:")
    for i, question in enumerate(questions, start=1):
        print(f"Question {i}: {question}")

if __name__ == "__main__":
    main()
