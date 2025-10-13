#pip install docxcompose

from docxcompose.composer import Composer
from docx import Document

files = ['Titulo.docx', 'examen.docx']  # tus archivos

master = Document(files[0])
composer = Composer(master)

for file in files[1:]:
    doc = Document(file)
    composer.append(doc)

composer.save('Inteligencia Artificial.docx')
