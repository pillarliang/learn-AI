import os
import cv2
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from paddle.utils import try_import

# 获取脚本所在目录的绝对路径
parent_dir = Path(__file__).resolve().parent.parent
print(parent_dir)
# 插入模块路径
sys.path.insert(0, str(parent_dir))
print(sys.path)

from paddleocr import PPStructure,save_structure_res, PaddleOCR, draw_structure_result
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx


ocr_engine = PPStructure(recovery=True, structure_version='PP-StructureV2')

save_folder = './layout_res'
pdf_path = '../data/basic.pdf'
font_path = '../fonts/simfang.ttf' # PaddleOCR下提供字体包

# 从 PDF 中获取页面图像
def covert_pdf_to_img(pdf_path):
    fitz = try_import("fitz")
    imgs = []

    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            pm = page.get_pixmap(alpha=False)
            mat = fitz.Matrix(2, 2) # 将 PDF 页面缩放两倍，为了提高图像的分辨率和清晰度。
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            # 将像素图转换为 PIL 图像对象。
            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples) 
            
            # 将 PIL 图像转换为 NumPy 数组，并从 RGB 格式转换为 BGR 格式，以便 OpenCV 使用。
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
            imgs.append(img)
            
    return imgs


if __name__ == "__main__":
    imgs = covert_pdf_to_img(pdf_path)
    print(len(imgs))