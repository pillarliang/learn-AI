import fitz

pdf_path = "../data/basic.pdf"
dpi = 300  # 设置所需的分辨率

with fitz.open(pdf_path) as pdf:
    page = pdf[0]  # 获取第一页
    rect = page.rect  # 获取页面的矩形区域
    width_points = rect.width
    height_points = rect.height

    width_pixels = int(width_points * (dpi / 72))
    height_pixels = int(height_points * (dpi / 72))

    print(f"Width: {width_pixels} pixels, Height: {height_pixels} pixels")
    