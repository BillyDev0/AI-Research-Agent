from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from TOOLS.web_search import web_search
from logger import logger
from AGENT.planner import planner
from typing import Literal
from pydantic import BaseModel

model = ChatOllama(
    model="qwen2.5:7b",
    top_p=0.1,
    temperature=0.2,
    num_thread=8,      # sesuaikan jumlah core CPU kamu
    num_ctx=2048,      # jangan set ctx lebih besar dari yang perlu — makin besar makin lambat
)

class Scema_output(BaseModel):
    status:Literal["complete","incomplete"]
    tools:Literal["web_search",""]
    query:str

structured_output=model.with_structured_output(Scema_output)

template = """
# ROLE
Kamu adalah Gap-Checker Agent. Tugasmu HANYA mengecek apakah data sudah cukup untuk menjawab GOAL_USER.

GOAL_USER:
{goal_user}

RIWAYAT_QUERY:
{riwayat_query}

DAFTAR_TOOLS:
{daftar_tools}

DATA_INFORMASI (data yang sudah terkumpul):
{jawaban}

# PRINSIP UTAMA: DEFAULT KE "complete"
Anggap status "complete" secara default, KECUALI ada alasan jelas dan spesifik untuk memilih "incomplete". Goal dianggap terjawab kalau intinya sudah kejawab, walau ada detail kecil yang belum sempurna.

JANGAN pilih "incomplete" hanya karena:
- Data terasa "bisa lebih detail" atau "bisa lebih lengkap".
- Kamu ingin menambah informasi yang TIDAK diminta di GOAL_USER.
- Suatu data sudah pernah dicari di RIWAYAT_QUERY tapi hasilnya memang tidak ketemu (anggap itu "tidak tersedia", BUKAN alasan untuk terus mencoba lagi).

Hanya pilih "incomplete" jika: ada entitas/aspek yang SECARA EKSPLISIT disebut di GOAL_USER, datanya BENAR-BENAR TIDAK ADA SAMA SEKALI di DATA_INFORMASI, DAN belum pernah dicoba dicari di RIWAYAT_QUERY.

#ALUR KERJA:
1. Cek DATA_INFORMASI terhadap PRINSIP UTAMA di atas.
2. Jika sudah memenuhi (default), isi field "status" dengan "complete", isi field lainnya seperti ini ("tools":"", "query":"").
3. Jika benar-benar belum memenuhi (sesuai syarat ketat di atas) -> isi field "status" dengan "incomplete", buat query untuk mencari SATU data yang belum ada, harus berhubungan langsung dengan GOAL_USER dan tidak boleh keluar dari context yang dibahas.
4. Field "tools" gunakan sesuai fungsinya yang ada di DAFTAR_TOOLS.

#KETENTUAN QUERY:
1. Query tidak boleh keluar dari konteks GOAL_USER.
2. Query HANYA boleh menyebut entitas yang PERSIS SAMA seperti tertulis di GOAL_USER — dilarang mengganti/menambah entitas lain (contoh: GOAL_USER sebut "iPhone 14 Pro" -> query tidak boleh jadi "iPhone 15" atau entitas lain).
3. Query hanya boleh mencari satu entitas+aspek secara spesifik seperti ("harga hp poco m6 pro", "chipset hp iPhone 14 pro", dll).
4. Query tidak boleh lebih dari 10 kata.
5. Query TIDAK BOLEH sama atau semirip makna dengan salah satu yang ada di RIWAYAT_QUERY.

# STATUS ITERASI
Sudah dilakukan {jumlah_iterasi_sekarang} kali pencarian dari maksimal {batas_iterasi} kali.
- Jika sudah lewat separuh batas_iterasi, jadilah LEBIH KETAT dalam memilih "incomplete".
- Jika sudah mendekati batas_iterasi (sisa 1-2 kali), WAJIB isi status "complete" apa pun kondisinya.

# FORMAT OUTPUT (WAJIB, JSON, TANPA TEKS LAIN)
{{
    "status": "...",
    "tools": "...",
    "query": "..."
}}
"""

def executor(plan):
    jawaban=[]
    goal_user=plan['goal']
    steps=plan['steps']
    riwayat_query=[]

    for data in steps:
        tools=data['tools']
        query=data['query']
        if "web_search" in tools:
            result=web_search(query)
            riwayat_query.append(query)
            
            [jawaban.append(results['content']) for results in result] 

    return jawaban,goal_user,riwayat_query

def agent_loop(jawaban,goal_user,riwayat_query,mapping_tools,max_iterasi=6):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|structured_output

    jumlah_iterasi=0
    for i in range(max_iterasi):
        result=chain.invoke({"goal_user":goal_user,
                             "riwayat_query":riwayat_query,
                             "daftar_tools":mapping_tools,
                             "jawaban":jawaban,
                             "jumlah_iterasi_sekarang":jumlah_iterasi,
                             "batas_iterasi":max_iterasi})
        jumlah_iterasi+=1
        status=result.status
        tools=result.tools
        query=result.query
        logger.info(f"RESULT ITERASI: {result}")

        if status=="complete":
            break

        elif status == "incomplete":

            if tools=="web_search":
                result=web_search(query)
                [jawaban.append(results['content']) for results in result]    
                riwayat_query.append(query)
                logger.info(f"Query: {query}")

        else:
            continue

    return jawaban


