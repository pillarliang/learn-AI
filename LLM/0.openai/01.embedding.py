import pandas as pd
import tiktoken
import os
import ast
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


current_path = os.path.dirname(os.path.abspath(__file__))

# openai embedding model parameters
client = OpenAI()
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# # ############# 1. Embedding #############
# load & inspect dataset
# to save space, we provide a pre-filtered dataset
input_dataPath = f"{current_path}/data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_dataPath, index_col=0)
print(df.head(2))
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
print(df.head(2))
# df = df.dropna()
# df["combined"] = "Title:" + df.Summary.str.strip() + ";Content" + df.Text.str.strip()
# print(df.head(2))

# top_n = 1000
# df = (
#     df.sort_values("Time").tail(top_n * 2)
# )
# df.drop("Time", axis=1, inplace=True)

# encoding = tiktoken.get_encoding(embedding_encoding)

# # omit reviews that are too long to embed
# df["n_tokens"] = df["combined"].apply(lambda x: len(encoding.encode(x)))
# df = df[df.n_tokens <= max_tokens].tail(top_n)

# # embedding model parameters
# client = OpenAI()


# def get_embedding(text, model=embedding_model):
#     text = text.replace("\n", " ")
#     return (
#         client.embeddings.create(
#             input=[text],
#             model=model,
#         )
#         .data[0]
#         .embedding
#     )


# df["embedding"] = df["combined"].apply(get_embedding)

# output_dataPath = f"{current_path}/data/fine_food_reviews_1k_embedding.csv"
# df.to_csv(output_dataPath)


# ############# 2. Visualizing the embeddings in 2D #############
# # We color the individual reviews based on the star rating which the reviewer has given
# embedding_dataPath = f"{current_path}/data/fine_food_reviews_with_embeddings_1k.csv"
# df_embedded = pd.read_csv(embedding_dataPath, index_col=0)
# df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# # # #turn embedding list into 2D numpy array
# matrix = np.array(df_embedded["embedding_vec"].tolist())
# # matrix = np.vstack(df_embedded["embedding_vec"].values)
# # # Create a t-SNE model and transform the data
# tsne = TSNE(
#     n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
# )
# vis_dims = tsne.fit_transform(matrix)

# colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
# x = [x for x, y in vis_dims]
# y = [y for x, y in vis_dims]

# color_indices = df_embedded.Score.values - 1
# colormap = matplotlib.colors.ListedColormap(colors)
# plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

# plt.title("Amazon ratings visualized in language using t-SNE")
# plt.show()


# #### 3. Application: K-means clustering of the embeddings ####
# from sklearn.cluster import KMeans

# embedding_dataPath = f"{current_path}/data/fine_food_reviews_with_embeddings_1k.csv"
# df_embedded = pd.read_csv(embedding_dataPath, index_col=0)
# df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)

# # turn embedding list into 2D numpy array
# matrix = np.vstack(df_embedded["embedding_vec"].values)

# n_clusters = 4
# kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
# kmeans.fit(matrix)
# df_embedded["Cluster"] = kmeans.labels_
# print(df_embedded.head(2))

# colors = ["red", "green", "blue", "purple"]
# tsne_model = TSNE(n_components=2, random_state=42)
# vis_data = tsne_model.fit_transform(matrix)

# x = vis_data[:, 0]
# y = vis_data[:, 1]

# color_indices = df_embedded.Cluster.values
# colormap = matplotlib.colors.ListedColormap(colors)
# plt.scatter(x, y, c=color_indices, cmap=colormap)
# plt.title("Clustering visualized in 2D using t-SNE")
# plt.show()


# ### 4. Text search using embeddings ###
# embedding_dataPath = f"{current_path}/data/fine_food_reviews_with_embeddings_1k.csv"
# df_embedded = pd.read_csv(embedding_dataPath, index_col=0)
# df_embedded["embedding_vec"] = df_embedded["embedding"].apply(ast.literal_eval)


# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# def embedding_text(text, model=embedding_model):
#     res = client.embeddings.create(input=text, model=model)
#     return res.data[0].embedding


# def search_reviews(df, product_description, n=3, pprint=True):
#     product_embedding = embedding_text(product_description)

#     df["similarity"] = df.embedding_vec.apply(
#         lambda x: cosine_similarity(x, product_embedding)
#     )

#     # combined："Title: Deceptive description; Content: On Oct 9 I ordered from a ..."
#     results = (
#         df.sort_values("similarity", ascending=False)
#         .head(n)
#         .combined.str.replace("Title: ", "")
#         .str.replace("; Content:", ": ")
#     )
#     if pprint:
#         for r in results:
#             print(r[:200])
#             print()
#     return results


# res = search_reviews(df_embedded, "delicious beans", n=3, pprint=True)
# print(res)
