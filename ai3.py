#!/usr/bin/env python3
import csv
import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# 加载CSV文件
file_path = os.path.abspath("/home/dev/langchain-ChatGLM/zc.csv")

questions = []
answers = []

with open(file_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        questions.append(row[0])
        answers.append(row[1])

# 定义文本分块的规则，这里用了一个很简单的规则，按照默认的分隔符来切割文本，使得每一段不超过1000个字符
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

# 把文本转换为向量。
embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")

# 建立向量索引
indexStore = Chroma.from_documents(answers, embeddings)
retriever = indexStore.as_retriever(search_kwargs={"k": 5})

while True:
    print("***********************************")
    query = input("Q：")
    result = retriever.get_relevant_documents(query)

    # 输出相似度前三的问题和对应的相似度
    for i, r in enumerate(result[:3]):
        print(f"A{i+1}: {questions[answers.index(r.page_content)]} (similarity: {r.score:.2f})")
