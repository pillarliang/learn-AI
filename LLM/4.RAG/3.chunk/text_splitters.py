from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)

with open(f"{CURRENT_DIR}/data/demo.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 100,
    chunk_overlap = 20,
    length_function=len,
    is_separator_regex = False,
)


text_splitter.split_text(state_of_the_union)