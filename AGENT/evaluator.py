from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from TOOLS.web_search import web_search 
from TOOLS.RAG_setup.RAG import similarity_search 
from logger import logger 
from typing import Literal 
from pydantic import BaseModel 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.2,
    top_p=0.1,
)

class Scema_output(BaseModel): 
    status:Literal["complete","incomplete"] 
    tools:Literal["web_search","","rag_search"] 
    query:str 
 
structured_output=model.with_structured_output(Scema_output) 
 
template = """
# ROLE
Kamu Evaluator Agent. Cek apakah DATA_INFORMASI cukup untuk GOAL_USER, cari tambahan jika kurang.

GOAL_USER: {goal_user}
RIWAYAT_QUERY: {riwayat_query}
DAFTAR_TOOLS: {daftar_tools}
DATA_INFORMASI: {jawaban}

# LANGKAH 1: BUANG SAMPAH
Buang bagian DATA_INFORMASI yang tidak membahas topik/syarat di GOAL_USER (produk lain, iklan, tidak relevan). Syarat mencakup: nama barang/topik, negara/wilayah, tahun, kategori, jumlah, atau batasan lain yang tertulis di GOAL_USER. Sisakan hanya yang relevan; jangan gunakan bagian yang dibuang di langkah selanjutnya.

# LANGKAH 2: BANDINGKAN DENGAN GOAL_USER
Baca sisa data secara wajar (bukan cocokkan kata per kata). Tanyakan: apakah semua yang diminta GOAL_USER sudah terjawab? Fokus hanya pada yang secara harfiah diminta, jangan mempertimbangkan info tambahan yang tidak diminta.

# LANGKAH 3: VALIDASI STATUS
"complete" jika: semua yang diminta sudah terjawab (makna sama walau beda kata), ATAU sudah pernah dicari (ada di RIWAYAT_QUERY) tapi tetap tidak ketemu (anggap tidak tersedia, jangan cari lagi).
"incomplete" jika: ada yang secara eksplisit diminta GOAL_USER, benar-benar tidak ada di data, DAN belum pernah dicoba di RIWAYAT_QUERY.
Pastikan sudah mencari secara maksimal sebelum memilih "complete".

# LANGKAH 4: VALIDASI QUERY (jika "incomplete")
Cek: query hanya sebut nama/topik PERSIS sama dengan GOAL_USER (jangan ganti/tambah entitas lain); hanya cari 1 jenis data yang harfiah diminta; beda makna dari semua RIWAYAT_QUERY; maksimal 10 kata, bentuk perintah. Jika status "complete", tools dan query dikosongkan.

# PEMILIHAN TOOLS
Pilih tool sesuai sumber data yang dibutuhkan, bukan berdasarkan nama. Info dari dokumen internal → tool akses dokumen internal. Info butuh internet → tool pencarian internet. Jangan buat nama tool baru.

# OUTPUT (JSON SAJA, TANPA TEKS LAIN)
{{"status": "...", "tools": "...", "query": "..."}}
"""
 
def executor(plan): 
    jawaban=[] 
    goal_user=plan['goal'] 
    steps=plan['steps'] 
    riwayat_query=[] 
 
    for data in steps: 
        tools=data['tools'] 
        query=data['query'] 
        if "web_search" == tools: 
            result=web_search(query) 
            [jawaban.append(results['content']) for results in result]  
 
        elif "rag_search" in tools: 
            result=similarity_search(query) 
            [jawaban.append(chunk) for chunk in result] 
            riwayat_query.append(query) 
             
 
    return jawaban,goal_user,riwayat_query 
 
def agent_loop(jawaban,goal_user,riwayat_query,mapping_tools,max_iterasi=4): 
    prompt=ChatPromptTemplate.from_template(template) 
    chain=prompt|structured_output 
 
    jumlah_iterasi=0 
    for i in range(max_iterasi): 
        if jumlah_iterasi==max_iterasi: 
            break 
        result=chain.invoke({"goal_user":goal_user, 
                             "riwayat_query":riwayat_query, 
                             "daftar_tools":mapping_tools, 
                             "jawaban":jawaban 
                            }) 
        status=result.status 
        tools=result.tools 
        query=result.query 
        logger.info(f"RESULT ITERASI: {result}") 
 
        if status=="complete": 
            break 
 
        elif status == "incomplete": 
            if query in riwayat_query: 
                continue 
 
            if tools=="web_search": 
                result=web_search(query) 
                [jawaban.append(results['content']) for results in result]     
                riwayat_query.append(query) 
                logger.info(f"Query: {query}") 
 
            elif tools=="rag_search": 
                break 
 
            jumlah_iterasi+=1 
 
 
        else: 
            continue 
 
    return jawaban