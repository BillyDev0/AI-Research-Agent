from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from AGENT.planner import planner

model=ChatOllama(model="qwen2.5:7b",
                top_p=0.3,
                temperature=0.5)


def tanya_AI(user_prompt):
    plan=planner(user_prompt)

    template = """
Plan:
{plan}

User:
{user_prompt}

Ikuti plan tersebut untuk menjawab user.
"""

    prompt=ChatPromptTemplate.from_template(template)
    chain=prompt|model
    result=chain.invoke({"plan":plan,"user_prompt":user_prompt})
    print(result.content)

prompt=input("masukan prompt: ")
tanya_AI(prompt)