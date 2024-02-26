import pandas as pd
import os
from openai import OpenAI


current_path = os.path.dirname(os.path.abspath(__file__))

# openai embedding model parameters
client = OpenAI()
embedding_model = (
    "text-embedding-ada-002"  # my openai api is not support for text-embedding-3-small
)
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

input_dataPath = f"{current_path}/data2/sheet1.csv"
df = pd.read_csv(input_dataPath)
df = df[["text"]]
df = df.dropna()


def get_embedding(text, model=embedding_model):
    return (
        client.embeddings.create(
            input=[text],
            model=model,
        )
        .data[0]
        .embedding
    )


df["embedding"] = df["text"].apply(get_embedding)  # 1536 dim
output_dataPath = f"{current_path}/data2/sheet1_embedding.csv"
df.to_csv(output_dataPath)
