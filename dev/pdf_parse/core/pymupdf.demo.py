import pymupdf4llm
import os

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

md_text = pymupdf4llm.to_markdown(f"{parent_path}/data/机器人-综合监控室.pdf", write_images=True)

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())