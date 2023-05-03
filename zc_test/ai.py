import csv
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
# from text2vec import Text2Vec
import faiss

# 设定向量维度
VECTOR_DIM = 300

# 读取csv问答文件的问题和答案
questions = []
answers = []
with open('/home/dev/langchain-ChatGLM/zc.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        # 跳过标题行
        if i == 0:
            continue
        # 将问题和答案加入列表
        questions.append(row[0])
        answers.append(row[1])

# 把文本转换为向量。
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

# 将问题和答案转换为向量
vectors_list = []
for text in (questions + answers):
    vector = embeddings.vectorize(text, size=VECTOR_DIM)
    vectors_list.append(vector)
vectors = np.array(vectors_list)

# 建立向量索引
index = faiss.IndexFlatIP(VECTOR_DIM)
index.add(vectors)

# 匹配相似度最高的答案
while True:
    query = input("Q：")
    query_vector = embeddings.vectorize(query, size=VECTOR_DIM)
    _, similar_ids = index.search(np.array([query_vector]), k=1)
    answer = answers[similar_ids[0][0]]

    # 输出相似度大于0的前三个问题和答案，以及对应的相似度
    _, similar_ids = index.search(vectors, k=4)
    print('相似度大于0的前三个问题和答案：')
    for i, similarity_list in enumerate(similar_ids):
        similar_questions = ', '.join([questions[s] for s in similarity_list[1:] if s != i])
        if similar_questions:
            print('\n原问题：', questions[i])
            print('相似问题：', similar_questions)
            print('原回答：', answers[i])
            for j, s in enumerate(similarity_list[1:]):
                if s != i:
                    print(f'相似问题{j+1}：{questions[s]}')
                    print(f'相似回答{j+1}：{answers[s]}')
                    print(f'相似度{j+1}：{index.search(np.array([vectors[s]]), k=1)[0][0]}')
        print('----------------------------------------')
