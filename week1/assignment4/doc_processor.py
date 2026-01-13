import docx
import PyPDF2
import os
 
def read_text_file(file_path: str):
    """Read content from a text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
 
def read_pdf_file(file_path: str):
    """Read content from a PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text
 
def read_docx_file(file_path: str):
    """Read content from a Word document"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
 
def read_document(file_path: str):
    """Read document content based on file extension"""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
 
    if file_extension == '.txt':
        return read_text_file(file_path)
    elif file_extension == '.pdf':
        return read_pdf_file(file_path)
    elif file_extension == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
 
def split_text(text: str, chunk_size: int = 500,overlap_size: int = 50):
    """Split text into chunks while preserving sentence boundaries"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
 
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
 
        # Ensure proper sentence ending
        if not sentence.endswith('.'):
            sentence += '.'
 
        sentence_size = len(sentence)
 
        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap = []
            total = 0

            for sentence in reversed(current_chunk):
                total += len(sentence)
                overlap.insert(0, sentence)
                if total >= overlap_size:
                    break
            current_chunk = overlap + [sentence]
            current_size = sum(len(s) for s in current_chunk)
     
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
 
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
 
    return chunks
 
 

 