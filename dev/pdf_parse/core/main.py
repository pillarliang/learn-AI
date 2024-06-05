import os
import cv2
import numpy as np
from paddleocr import PPStructure,save_structure_res, PaddleOCR, draw_structure_result
from paddle.utils import try_import
from PIL import Image
from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

ocr_engine = PPStructure(recovery=True, structure_version='PP-StructureV2')

save_folder = './dev_v1'
pdf_path = '../data/基础版.pdf'
font_path = '../fonts/simfang.ttf' # PaddleOCR下提供字体包


fitz = try_import("fitz")
imgs = []
with fitz.open(pdf_path) as pdf:
    for pg in range(0, pdf.page_count):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2) # 将 PDF 页面缩放两倍，为了提高图像的分辨率和清晰度。
        pm = page.get_pixmap(matrix=mat, alpha=False)

        # if width or height > 2000 pixels, don't enlarge the image
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples) # 将像素图转换为 PIL 图像对象。
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # 将 PIL 图像转换为 NumPy 数组，并从 RGB 格式转换为 BGR 格式，以便 OpenCV 使用。
        imgs.append(img)
        


if __name__ == "__main":
    # TODO: 加载PDF页面，传入模型进行处理
    pages = get_pdf_page(pdf_path)
    
    model(page)
    
    
    
    
    