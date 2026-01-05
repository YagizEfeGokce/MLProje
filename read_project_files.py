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

files = [
    "COE305 Machine Learning Project Guidelines.pdf",
    "Stage 1 Project(Done).docx",
    "Stage 3 Project template.docx"
]

for filename in files:
    print(f"\n{'='*20}\nFILE: {filename}\n{'='*20}")
    if not os.path.exists(filename):
        print("File not found.")
        continue
    
    if filename.lower().endswith(".pdf"):
        content = read_pdf(filename)
    elif filename.lower().endswith(".docx"):
        content = read_docx(filename)
    else:
        content = "Unsupported format"
    
    # Print first 2000 chars to avoid huge output, but enough to verify reading
    print(content[:2000])
    if len(content) > 2000:
        print(f"\n... (Truncated, total length: {len(content)} chars)")
