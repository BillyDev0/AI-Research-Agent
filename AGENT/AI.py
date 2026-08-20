from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from AGENT.planner import planner
from TOOLS.web_search import web_search
from AGENT.evaluator import agent_loop,executor
from AGENT.extractor import extractor
from logger import logger
import asyncio

model = ChatOllama(
    model="qwen2.5:7b",
    top_p=0.2,
    temperature=0.3,
    num_thread=8,      # sesuaikan jumlah core CPU kamu
    num_ctx=2048,      # jangan set ctx lebih besar dari yang perlu — makin besar makin lambat
)

jawaban=["""HP A memiliki harga Rp3.000.000 dan RAM 8GB.
HP B memiliki harga Rp4.000.000 dan RAM 12GB."""]
async def tanya_AI(user_prompt):
    # plan=planner(user_prompt)

    # jawaban,riwayat_query=executor(plan)
    # logger.info(f"HASIL SEARCH: {jawaban}")

    evaluator=agent_loop(user_prompt,jawaban)
    logger.info(f"HASIL EVALUATOR: {evaluator}")
  
    # result_final=extractor(user_prompt,evaluator)
    template="""
Kamu adalah AI yang bertugas menyusun jawaban akhir berdasarkan data hasil pencarian.

ATURAN WAJIB:

1. Jawab HANYA permintaan yang ditulis oleh user.
2. Jangan membuat pertanyaan baru.
3. Jangan membuat sub-pertanyaan yang tidak diminta user.
4. Jangan memberikan rekomendasi tambahan yang tidak diminta.
5. Jangan menggunakan pengetahuan dari luar DATA.
6. Jangan mengarang informasi yang tidak terdapat dalam DATA.
7. Jika suatu informasi tidak tersedia dalam DATA, tulis "Data tidak tersedia".
8. Gunakan hanya data yang relevan dengan kebutuhan user.
9. Jangan menampilkan data yang tidak berhubungan dengan pertanyaan user.
10. Jika user meminta perbandingan, buat perbandingan yang langsung sesuai dengan kriteria yang diminta.
11. Jika user meminta 3 item, tampilkan tepat 3 item.
12. Jika user meminta sumber, sertakan sumber yang tersedia di DATA.

Jika data mengandung LEBIH BANYAK item daripada jumlah_item yang diminta:
1. Jangan langsung ambil N item pertama secara acak.
2. Rangking semua item kandidat berdasarkan relevansi dengan kriteria user
   (contoh: kalau user prioritaskan "performa", urutkan berdasarkan itu).
3. Ambil TOP jumlah_item saja.
4. Item yang tidak terpilih TIDAK ditampilkan sama sekali di final_answer,
   termasuk tidak disebut sebagai "opsi tambahan" atau "bonus".

KEBUTUHAN USER:
{user_prompt}

DATA HASIL PENCARIAN:
{result_tools}

TUGAS:

Buat jawaban langsung untuk kebutuhan user.

Jangan menjelaskan proses pencarian.
Jangan menjelaskan proses berpikir.
Jangan membuat pertanyaan baru.

Jawaban:"""

    try:
        prompt=ChatPromptTemplate.from_template(template)
        chain=prompt|model
        for chunk in chain.stream({"user_prompt":user_prompt, "result_tools":evaluator}):
            print(chunk.content, end="", flush=True)
                
        print()
           
    except Exception as e:
        logger.exception(e)
        
prompt=input("masukan prompt: ")
asyncio.run(tanya_AI(prompt))