import pandas as pd
# from text2vec.models import FastText
from text2vec.vectorizers import Vectorizer
import numpy as np
from annoy import AnnoyIndex

# 加载 text2vec-large-chinese 模型
model_path = "/home/dev/text2vec-large-chinese"
# model = FastText.load_fasttext_format(model_path)

# 加载问答数据
data = pd.read_csv("/home/dev/langchain-ChatGLM/dtl.csv")

# 将问题和答案转换为向量
vectorizer = Vectorizer.load(model_path)
vectors = vectorizer.transform(data["question"] + data["answer"])

# 将向量保存到 numpy 数组中
vectors_array = np.vstack(list(vectors))

# 建立向量索引
index = AnnoyIndex(vectors_array.shape[1], "angular")
for i, vector in enumerate(vectors_array):
    index.add_item(i, vector)
index.build(10)


# 定义一个函数，根据问题匹配相似度最高的答案
def find_answer(question):
    # 将输入的问题转换为向量
    question_vector = vectorizer.transform([question])[0]

    # 使用 annoy 索引查找相似的向量
    ids, distances = index.get_nns_by_vector(question_vector, 3, include_distances=True)

    # 打印相似度大于 0 的前三个问题和答案，以及对应的相似度
    for id, distance in zip(ids, distances):
        if distance > 0:
            print("Question: ", data.iloc[id]["question"])
            print("Answer: ", data.iloc[id]["answer"])
            print("Similarity: ", 1 - distance)
            print()


while True:
    query = input("Q：")
    find_answer(query)
