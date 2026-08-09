from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import json

model=ChatOllama(model="qwen2.5:7b",
                top_p=0.3,
                temperature=0.5)


daftar_action=[

    {
        "action":"web_search",
        "description":"Mencari informasi terkini di internet, gunakan untuk pertanyaan yang butuh data baru/real-time"
    },

]

template = """Kamu adalah AI Planner.

Tugasmu HANYA membuat rencana (plan) untuk melaksanakan perintah user di bawah ini.
Jangan menjawab, menjalankan, atau menyelesaikan perintah tersebut — cukup pecah jadi langkah-langkah yang actionable.

Perintah user:
<user_request>
{user_prompt}
</user_request>

Aturan:
- Jawab HANYA dalam format JSON valid, tanpa teks tambahan, tanpa markdown code fence (```).
- Bahasa di dalam "goal" dan "steps" mengikuti bahasa perintah user.
- Setiap step harus konkret dan actionable (bukan penjelasan umum).
- Field "action" WAJIB salah satu dari: {daftar_action}
- Jika perintah user ambigu, buat asumsi paling wajar dan sebutkan asumsinya sebagai step pertama.
- jika tidak ada action yang sesuai/cocok dengan perintah user, you can say "Tools tidak tersedia"

Format output:
{{
    "goal": "...",
    "steps": [
        {{
           "action":"...",
           "query":"..."
        }}
    ]
}}
"""

def planner(user_prompt):
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model
    result=chain.invoke({"user_prompt":user_prompt,"daftar_action":daftar_action})
    plan=json.loads(result.content)
    return plan['steps']
