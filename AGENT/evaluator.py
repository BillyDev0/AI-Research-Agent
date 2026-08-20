from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from TOOLS.web_search import web_search
from logger import logger
import json
from pydantic import BaseModel
from typing import Literal


model = ChatOllama(
    model="qwen2.5:7b",
    top_p=0.1,
    temperature=0.2,
    num_thread=8,      # sesuaikan jumlah core CPU kamu
    num_ctx=2048,      # jangan set ctx lebih besar dari yang perlu — makin besar makin lambat
)

class EvaluatorOutput(BaseModel):
    keputusan:Literal["final","web_search"]
    query:str

structured_output=model.with_structured_output(EvaluatorOutput)

daftar_tools=[

    {
        "action":"web_search",
        "description":"Mencari informasi terkini di internet, gunakan untuk pertanyaan yang butuh data baru/real-time"
    },

]

template = """
Kamu adalah Data Completeness Checker.

Tugasmu HANYA menentukan apakah semua informasi yang diminta USER sudah tersedia di DATA.

USER:
{user_prompt}

DATA:
{data_jawaban}

RIWAYAT QUERY:
{riwayat_query}

Ikuti langkah berikut:

1. Baca USER.
2. Buat daftar semua informasi yang secara eksplisit diminta USER.
3. Untuk SETIAP informasi tersebut, cari apakah informasinya tersedia di DATA.
4. Jika SATU SAJA informasi yang diminta tidak tersedia, keputusan HARUS "web_search".
5. Jika SEMUA informasi tersedia, keputusan "final".
6. Jangan menilai kualitas, kedalaman, confidence, atau relevansi. Fokus HANYA pada kelengkapan data.
7. Jangan menggunakan pengetahuan dari luar DATA.
8. Jika informasi tidak disebutkan secara eksplisit di DATA, anggap informasi tersebut BELUM TERSEDIA.
9. Jika keputusan "web_search", query harus menyebutkan informasi yang belum tersedia.
10. Jika keputusan "final", query harus "".

## ENTITY-FIELD CHECK

Cek setiap kombinasi entitas dan field yang diminta user.

Contoh:
User: "Bandingkan HP A dan HP B dari harga, RAM, dan baterai."

Wajib ada:
HP A → harga, RAM, baterai
HP B → harga, RAM, baterai

Informasi baterai dari HP lain tidak memenuhi kebutuhan HP A atau HP B.

Jika ada satu saja kombinasi yang belum tersedia → "web_search".
Jika semua tersedia → "final".


## SEARCH HISTORY

SEARCH_HISTORY berisi query yang sudah pernah digunakan.

Jangan gunakan kembali query yang sama atau memiliki makna yang sama.

Jika data masih belum lengkap:
- buat query baru yang berbeda dari {riwayat_query}
- query baru harus lebih spesifik terhadap data yang masih kurang

OUTPUT:
{{
  "keputusan": "final" atau "web_search",
  "query": "..."
}}


"""

def executor(plan):
    logger.info(plan)

    jawaban=[]
    query=[]
    for step in plan:
        if "web_search" in step['action']:
            results=web_search(step['query'])
            query.append(step['query'])

            for result in results:
                jawaban.append(result['content'])

    return jawaban,query

def agent_loop(user_prompt,data_jawaban):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|structured_output

    riwayat_query=[]
    max_iteration=4
    while True:
        for iteration in range(max_iteration):
            keputusan=chain.invoke({"user_prompt":user_prompt,"data_jawaban":data_jawaban,"riwayat_query":riwayat_query})
            logger.info(f"KEPUTUSAN: {keputusan}")

            action=keputusan.keputusan
            query=keputusan.query

            if action == "web_search":
                results=web_search(query)
                riwayat_query.append(query)
                
                for result in results:
                    data_jawaban.append(result['content'])

            elif action == "final":
                break

            else:
                continue


        return data_jawaban
    
