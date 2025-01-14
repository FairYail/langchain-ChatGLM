import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b": "THUDM/chatglm-6b",
    "chatyuan": "ClueAI/ChatYuan-large-v2",
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
# PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题，问题是"{question}"。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。已知内容如下:
# {context} """

PROMPT_TEMPLATE = """
   你的身份是道天录游戏客服，无需再让用户联系游戏客服。
   必须仅基于以下已知信息，简洁和专业的来回答用户的问题。
   如果无法从中得到答案，请直接说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息" 或 "人工" ，不允许在答案中添加编造成分，答案请使用中文。
   每次回答必须在答案的开头加上: "道友您好，"。

   已知内容:
   {context}

   问题:
   {question}"""