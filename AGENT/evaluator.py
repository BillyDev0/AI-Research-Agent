from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from TOOLS.web_search import web_search
from TOOLS.RAG_setup.RAG import similarity_search
from logger import logger
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
    tools:Literal["web_search","","rag_search"]
    query:str

structured_output=model.with_structured_output(Scema_output)

template = """
# ROLE
Kamu adalah Evaluator Agent. Tugasmu mengecek apakah DATA_INFORMASI sudah cukup untuk menjawab GOAL_USER, dan mencari data tambahan jika masih kurang.

GOAL_USER:
{goal_user}

RIWAYAT_QUERY:
{riwayat_query}

DAFTAR_TOOLS:
{daftar_tools}

DATA_INFORMASI (hasil pencarian, BISA MENGANDUNG DATA TIDAK RELEVAN):
{jawaban}

# LANGKAH 1: BUANG SAMPAH
Baca DATA_INFORMASI, abaikan/buang total bagian yang TIDAK membahas topik/barang yang disebut di GOAL_USER (produk lain, iklan, artikel tidak nyambung, dll). Sisakan hanya bagian yang benar-benar tentang topik di GOAL_USER. Jangan pernah menyebut atau menggunakan isi yang sudah dibuang ini di langkah manapun setelah ini.
aca GOAL_USER dan identifikasi SEMUA syarat yang disebutkan di dalamnya — bisa berupa: nama barang/topik utama, DAN/ATAU syarat tambahan seperti negara, wilayah, tahun, jangka waktu, kategori, jumlah, kondisi, atau batasan lain apa pun yang tertulis di GOAL_USER.

# LANGKAH 2: BANDINGKAN DENGAN GOAL_USER
Baca sisa data (yang sudah bersih) dengan santai, seperti manusia membaca artikel biasa. Tanyakan: "Apakah semua yang diminta di GOAL_USER sudah terjawab oleh data ini?" Fokus HANYA pada apa yang secara harfiah diminta GOAL_USER — jangan mempertimbangkan informasi tambahan yang tidak diminta.

# LANGKAH 3: VALIDASI STATUS
Pilih "complete" JIKA:
- Semua hal yang diminta GOAL_USER sudah terjawab oleh data bersih (walau kalimatnya tidak persis sama seperti di GOAL_USER, cukup maknanya sama), ATAU
- Ada bagian yang sudah dicoba dicari (ada di RIWAYAT_QUERY) tapi tetap tidak ditemukan — anggap itu "tidak tersedia", bukan alasan mencari lagi.

Pilih "incomplete" JIKA:
- Ada bagian yang SECARA EKSPLISIT diminta GOAL_USER, BENAR-BENAR TIDAK ADA sama sekali di data bersih, DAN belum pernah dicoba dicari di RIWAYAT_QUERY.

sebelum menghasilkan status "complete" harus dipastikan mencari entitas secara maksimal (bukan hanya mengandalkan data bersih).

# LANGKAH 4: VALIDASI QUERY (HANYA JIKA STATUS "incomplete")
Sebelum menulis query, cek satu per satu:
- Apakah query ini HANYA menyebut nama barang/topik yang PERSIS SAMA dengan yang ada di GOAL_USER? (dilarang mengganti/menambah nama lain, walau sejenis/semerek)
- Apakah query ini HANYA mencari 1 jenis data yang SECARA HARFIAH diminta GOAL_USER? (dilarang mencari hal lain yang terasa relevan tapi tidak diminta)
- Apakah query ini BEDA maknanya dari semua RIWAYAT_QUERY?
- Apakah query ini singkat (maksimal 10 kata) dan berbentuk perintah?

Jika salah satu jawaban di atas "tidak", perbaiki dulu query sebelum dijadikan output. Jika status "complete", tools dan query dikosongkan.

# PEMILIHAN TOOLS

Pilih tool berdasarkan deskripsi tool dan kebutuhan setiap step.

- Gunakan tool yang sumber datanya paling sesuai dengan informasi yang diminta.
- Jangan memilih tool hanya berdasarkan nama.
- Jika informasi yang diminta berasal dari dokumen internal, gunakan tool yang menyediakan akses ke dokumen internal.
- Jika informasi membutuhkan internet, gunakan tool yang menyediakan pencarian internet.
- Jangan membuat nama tool baru.

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
        if "web_search" == tools:
            result=web_search(query)
            riwayat_query.append(query)
            
            [jawaban.append(results['content']) for results in result] 

        elif "rag_search" in tools:
            result=similarity_search(query)
            jawaban.append(result.page_content)
            

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


