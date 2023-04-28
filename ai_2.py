import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取 CSV 文件
with open('/home/dev/langchain-ChatGLM/zc.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)

# 提取问题和答案
questions = [row[0] for row in data[1:]]
answers = [row[1] for row in data[1:]]

# 生成 TF-IDF 特征向量
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(questions)


# 定义问答匹配函数
def match_query(query, questions, tfidf_vectors):
    # 生成查询向量
    query_vector = vectorizer.transform([query])
    # 计算余弦相似度
    similarities = cosine_similarity(query_vector, tfidf_vectors).flatten()
    # 获取相似度最高的问题及答案
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    max_question = questions[max_idx]
    max_answer = answers[max_idx]
    return max_sim, max_question, max_answer


# 测试匹配函数
while True:
    query = input("Q：")
    sim, question, answer = match_query(query, questions, tfidf_vectors)
    print(f"查询：{query}")
    print(f"匹配问题：{question}")
    print(f"匹配答案：{answer}")
    print(f"相似度：{sim}")
