import glob
from PyPDF2 import PdfMerger

files = glob.glob('open/*.pdf')
merger = PdfMerger()

for pdf in files:
    merger.append(pdf)
    print(f"Agregado: {pdf}")

merger.write('2020_output.pdf')
merger.close()
print("PDFs concatenados correctamente.")
