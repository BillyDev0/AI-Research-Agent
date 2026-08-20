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

template="""
Kamu adalah Information Extractor.

Tugas:
Ambil informasi yang diminta USER dari DATA.

USER:
{prompt}

DATA:
{data_jawaban}

ATURAN:
1. Ambil HANYA informasi yang diminta USER.
2. Jangan menggunakan pengetahuan dari luar DATA.
3. Jika informasi tidak ditemukan, isi null.
4. Jangan mengarang atau menyimpulkan nilai.
5. Jika USER menyebut beberapa item, proses semuanya.
6. Output HARUS JSON valid.
7. Jangan memberikan penjelasan tambahan.

FORMAT:
[
  {{
    "nama_item": "...",
    "data": {{
      "field": "value"
    }}
  }}
]
"""

def extractor(user_prompt,result_tools):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model
    result=chain.invoke({"prompt":user_prompt,"data_jawaban":result_tools})
    logger.info(result)

    result=json.loads(result.content)
    logger.info(f"HASIL EXTRACTOR: {result}")
    return result