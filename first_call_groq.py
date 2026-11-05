from dotenv import load_dotenv
from groq import Groq 

load_dotenv()

client = Groq() 

context = """Model Context Protocol (MCP) is an open standard that lets AI
applications connect to external tools and data sources in a uniform way.
Instead of writing custom integration code for every tool, a developer
exposes a tool through an MCP server, and any MCP-compatible AI app
(the client) can discover and use it."""

question = "Who created MCP and in which year?"

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": "Answer only from the context given. If the context "
                       "does not contain the answer, say you don't know.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ],
)

print(response.choices[0].message.content)
