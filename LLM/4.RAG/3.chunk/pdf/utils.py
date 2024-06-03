import PyPDF2
import os
import subprocess

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_pdf_type(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = reader.pages[0].extract_text()
            if text:
                return True
            else:
                return False
    except Exception as e:
        print(f"error occurs when open pdf, {e}")
        
        
        
def check_if_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        return True if result else False
    except FileNotFoundError:
        return False
        
        
if __name__ == '__main__':
    pdf_path = f'{CURRENT_DIR}/data/image-based.pdf'
    res = check_pdf_type(pdf_path)
    # print(res)
    # print(check_if_gpu())
    