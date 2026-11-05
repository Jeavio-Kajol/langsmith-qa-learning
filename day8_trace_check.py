from dotenv import load_dotenv
from langsmith import Client, traceable
import os

load_dotenv()

print("Tracing enabled:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))
print("API key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))

client = Client()

@traceable(
    name="day8-tracing-check",
    client=client
)
def test_tracing():
    return "LangSmith tracing test successful"

result = test_tracing()

print("Result:", result)

client.flush()

print("Tracing test completed.")