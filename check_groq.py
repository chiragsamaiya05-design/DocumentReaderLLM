import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    models = client.models.list()

    print("\nAvailable models:\n")

    for model in models.data:
        if "gpt-oss" in model.id:
            print(model.id)

except Exception as error:
    print("Groq connection failed:", error)