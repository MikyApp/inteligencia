import fitz  # PyMuPDF para leer PDFs
from openai import OpenAI

# Función para extraer texto de un solo PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Función para extraer y concatenar texto de varios PDFs
def extract_texts_from_pdfs(pdf_paths):
    combined_text = ""
    for path in pdf_paths:
        combined_text += extract_text_from_pdf(path) + "\n\n"
    return combined_text

# Configurar cliente Ollama (tipo OpenAI API)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Lista con todos los PDFs que quieres incluir
pdf_files = ["Semana4.pdf", "Semana5.pdf", "Semana6.pdf", "Semana7.pdf"]

# Extraer texto combinado de todos los PDFs
pdf_text = extract_texts_from_pdfs(pdf_files)

pregunta = input("Haz preguntas al modelo: ")
prompt = f"A continuación se presenta el contenido de varios documentos:\n{pdf_text}\n{pregunta}"

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "Eres un asistente experto que responde basado en el texto proporcionado."},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
