from groq import Groq 
from dotenv import load_dotenv

load_dotenv() 

MODEL = "openai/gpt-oss-120b"
client = Groq()

def ask_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

