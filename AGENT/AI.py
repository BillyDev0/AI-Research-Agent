from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from AGENT.planner import planner
from TOOLS.web_search import web_search


def executor(plan):
    jawaban=[]
    for step in plan:
        if "web_search" in step['action']:
            results=web_search(step['query'])
            for result in results:
                jawaban.append(result['content'])
            
    return "\n".join(jawaban)


model=ChatOllama(model="qwen2.5:7b",
                top_p=0.3,
                temperature=0.5)

def tanya_AI(user_prompt):
    plan=planner(user_prompt)
    result_tools=executor(plan)
    template = """
    anda adalah seorang Asisten spesialis search 

Rules:
1. Kamu HANYA boleh menjawab berdasarkan Data hasil pencarian.
2. Jika data berisi beberapa informasi, rangkum dengan bahasa yang jelas.
3. Jawaban harus sepenuhnya berasal dari Data hasil pencarian.
4. Jangan menggunakan pengetahuanmu sendiri.

User:
{user_prompt}

Data hasil pencarian:
{result_tools}

Jawab pertanyaan user berdasarkan data di atas.
"""
    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model
    result=chain.invoke({"user_prompt":user_prompt, "result_tools":result_tools})

    print("OUTPUT AI AGENT")
    print(result.content)

prompt=input("masukan prompt: ")
tanya_AI(prompt)