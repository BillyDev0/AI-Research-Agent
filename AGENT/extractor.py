from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import json
from logger import logger
model = ChatOllama(
    model="qwen2.5:7b",
    top_p=0.2,
    temperature=0.3,
    num_thread=8,      # sesuaikan jumlah core CPU kamu
    num_ctx=2048,      # jangan set ctx lebih besar dari yang perlu — makin besar makin lambat
)

template = """ JAWAB PERTANYAAN USER DENGAN PENGETAHUANMU 
USER PROMPT:
{user_prompt}
"""


def extractor(prompt_user):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model
    result=chain.invoke({prompt_user})

    result=result.content
    return result

