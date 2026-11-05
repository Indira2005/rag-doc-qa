from dotenv import load_dotenv
from google import genai 

load_dotenv() 

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="What is MCP in AI? Explain in short and simple way."
)

print(response.text)