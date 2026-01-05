import os
import pypdf
import docx

def read_pdf(path):
    try:
        reader = pypdf.PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def read_docx(path):
    try:
        doc = docx.Document(path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        return f"Error reading DOCX: {e}"

files = {
    "COE305 Machine Learning Project Guidelines.pdf": "guidelines_content.txt",
    "Stage 1 Project(Done).docx": "stage1_content.txt",
    "Stage 3 Project template.docx": "stage3_template_content.txt"
}

for filename, output_name in files.items():
    print(f"Processing {filename}...")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    
    if filename.lower().endswith(".pdf"):
        content = read_pdf(filename)
    elif filename.lower().endswith(".docx"):
        content = read_docx(filename)
    else:
        content = "Unsupported format"
    
    with open(output_name, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to {output_name}")
