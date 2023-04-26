from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from models.chatglm_llm import ChatGLM
import sentence_transformers
import os
from configs.model_config import *
import datetime
from typing import List
from textsplitter import ChineseTextSplitter
from langchain.docstore.document import Document

# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6

# LLM input history length
LLM_HISTORY_LEN = 3


def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


class LocalDocQA:
    llm: object = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 llm_model: str = LLM_MODEL,
                 llm_device=LLM_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 use_ptuning_v2: bool = USE_PTUNING_V2
                 ):
        self.llm = ChatGLM()
        self.llm.load_model(model_name_or_path=llm_model_dict[llm_model],
                            llm_device=llm_device,
                            use_ptuning_v2=use_ptuning_v2)
        self.llm.history_len = llm_history_len

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    print(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        print(f"{file} 已成功加载")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} 未能成功加载")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    print(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} 未能成功加载")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    # def get_knowledge_based_answer(self,
    #                                query,
    #                                vs_path,
    #                                chat_history=[],
    #                                streaming=True):
    #     self.llm.streaming = streaming
    #     vector_store = FAISS.load_local(vs_path, self.embeddings)
    #     related_docs_with_score = vector_store.similarity_search_with_score(query,
    #                                                                         k=self.top_k)
    #     related_docs = get_docs_with_score(related_docs_with_score)
    #     prompt = generate_prompt(related_docs, query)
    #
    #     if streaming:
    #         for result, history in self.llm._call(prompt=prompt,
    #                                               history=chat_history):
    #             history[-1][0] = query
    #             response = {"query": query,
    #                         "result": result,
    #                         "source_documents": related_docs}
    #             yield response, history
    #     else:
    #         result, history = self.llm._call(prompt=prompt,
    #                                          history=chat_history)
    #         history[-1][0] = query
    #         response = {"query": query,
    #                     "result": result,
    #                     "source_documents": related_docs}
    #         return response, history

    # def get_knowledge_based_answer(self,
    #                                query,
    #                                vs_path,
    #                                chat_history=[], ):
    #     prompt_template = """
    # 你的身份是道天录游戏客服，无需再让用户联系游戏客服。
    # 必须仅基于以下已知信息，简洁和专业的来回答用户的问题。
    # 如果无法从中得到答案，请直接说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文或数字或符号。
    # 每次回答必须在答案的开头加上: "道友您好，"。
    #
    # 已知内容:
    # {context}
    #
    # 问题:
    # {question}"""
    #     prompt = PromptTemplate(
    #         template=prompt_template,
    #         input_variables=["context", "question"]
    #     )
    #     self.llm.history = chat_history
    #     vector_store = FAISS.load_local(vs_path, self.embeddings)
    #     knowledge_chain = RetrievalQA.from_llm(
    #         llm=self.llm,
    #         retriever=vector_store.as_retriever(search_kwargs={"k": self.top_k}),
    #         prompt=prompt
    #     )
    #     knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
    #         input_variables=["page_content"], template="{page_content}"
    #     )
    #
    #     knowledge_chain.return_source_documents = True
    #
    #     result = knowledge_chain({"query": query})
    #     self.llm.history[-1][0] = query
    #     return result, self.llm.history

    # 结合知识库进行问题回答
    def get_knowledge_based_answer(self, query, vector_store, chat_history=[]):
        global chatglm, embeddings

        prompt_template = """
        你的身份是道天录游戏客服。
        必须仅基于知识库信息，简洁和专业的来回答用户的问题。
        如果无法从中得到答案，请直接说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息" 或 "人工客服"，不允许在答案中添加编造成分，答案请使用中文或数字或符号。
        每次回答必须在答案的开头加上: "道友您好，"。

        已知内容:
        {context}

        问题:
        {question}"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        chatglm.history = chat_history
        knowledge_chain = RetrievalQA.from_llm(
            llm=chatglm,
            retriever=vector_store.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
            prompt=prompt
        )
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        chatglm.history[-1][0] = query
        return result, chatglm.history
