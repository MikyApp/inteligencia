from docx import Document

def docx_to_txt(docx_path, txt_path):
    document = Document(docx_path)
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        for para in document.paragraphs:
            txt_file.write(para.text + '\n')

# Uso:
docx_to_txt('Inteligencia.docx', 'documento.txt')


#Responde todas la preguntas del documento.txt