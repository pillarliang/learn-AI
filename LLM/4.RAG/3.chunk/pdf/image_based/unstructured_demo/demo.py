from unstructured.partition.pdf import partition_pdf
import os

CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  
file_path = f'{CURRENT_DIR}/data/测试.pdf'
  
# infer_table_structure=True automatically selects hi_res strategy  
elements = partition_pdf(filename=file_path, infer_table_structure=True, languages=["chi_sim"], chunking_strategy='by_title')

for ele in elements:
    print(ele.text)