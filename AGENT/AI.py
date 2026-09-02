from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from AGENT.planner import planner
from AGENT.evaluator import executor,agent_loop
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

template="""
# ROLE
Kamu adalah Presenter Agent. Tugasmu HANYA menyusun ulang data yang sudah ada menjadi jawaban untuk user. Kamu TIDAK mencari data baru dan TIDAK menambahkan informasi apa pun di luar data yang diberikan.

# ATURAN
1. HANYA gunakan informasi dari EXECUTION_RESULTS. DILARANG menambahkan fakta, angka, atau klaim yang tidak ada di dalamnya — walau kamu tahu jawabannya dari pengetahuan sendiri.
2. Jika ada bagian goal yang tidak terjawab oleh EXECUTION_RESULTS, katakan terus terang bahwa informasinya tidak ditemukan. Jangan menebak.
3. Sertakan sumber jika tersedia di data.
4. Jawaban mengikuti bahasa yang user gunakan.
5. Susun jawaban dengan rapi (poin/tabel/paragraf singkat) sesuai bentuk data.

GOAL_USER:
{goal_user}

EXECUTION_RESULTS:
{result}

# OUTPUT
Tulis jawaban akhir untuk user berdasarkan EXECUTION_RESULTS di atas, tanpa menambah data apa pun.
"""

mapping_tools=[
    
    {
        "name":"web_search",
        "descriptions":"tools ini digunakan jika ingin mencari/search informasi di internet"
    }
]

async def tanya_AI(user_prompt):
    plan=planner(user_prompt,mapping_tools)
    logger.info(f"RESULT PLANNER: {plan}")

    executor_result,goal_user,riwayat_query=executor(plan)
    logger.info(f"RESULT EXECUTOR: {executor_result}")

    evaluator_result=agent_loop(executor_result,goal_user,riwayat_query,mapping_tools)
    logger.info(f"RESULT EVALUATOR: {evaluator_result}")
    
    try:
        prompt=ChatPromptTemplate.from_template(template)
        chain=prompt|model
        for chunk in chain.stream({"goal_user":goal_user,"result":evaluator_result}):
            print(chunk.content, end="", flush=True)
            await asyncio.sleep(0.05)
                
        print()
           
    except Exception as e:
        logger.exception(e)

prompt=input("prompt: ")
asyncio.run(tanya_AI(prompt))