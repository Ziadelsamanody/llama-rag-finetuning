from model import generate_rag, load_model
from vectorstore import VectorStore
from langchain_community.chat_message_histories import ChatMessageHistory
import os

vs = VectorStore()

if not os.path.exists("index/index.faiss"):
    print("Error: Vector store not found. Please run 'python ingest.py' first.")
    exit(1)

load_model()
vs.load()

history = ChatMessageHistory()

print("RAG Chat ready! (type 'quit' to exit)\n")

while True:
    query = input("You: ").strip()
    if query.lower() in {"quit", "exit", "q"}:
        break
    if not query:
        continue

    context = vs.search(query, k=3)
    
    recent = history.messages[-4:]
    history_list = [
        {"role": msg.type, "content": msg.content} for msg in recent
    ]

    answer = generate_rag(context, history_list, query)
    print("Assistant:", answer, "\n")

    history.add_user_message(query)
    history.add_ai_message(answer)

