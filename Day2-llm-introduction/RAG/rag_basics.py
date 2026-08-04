# RAG for Beginners – Step by Step
# Use case: Campus FAQ bot
#
# Setup (run once in terminal):
#   pip install ollama
#   ollama pull llama3.2
#   ollama pull nomic-embed-text
#
# Copy each CELL into Jupyter and run one by one.


# %% CELL 0: Setup

from ollama import chat, embed
import math

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

print("Setup done.")


# %% CELL 1: Step 1 – What is RAG?

print("""
========== What is RAG? ==========

  Question
     |
     v
  Find related notes   <--- RETRIEVE
     |
     v
  Add notes to prompt  <--- AUGMENT
     |
     v
  LLM gives answer     <--- GENERATE

We answer ONLY from our notes. If notes don't have it → "I don't know"
==================================
""")


# %% CELL 2: Step 2 – Create the knowledge base

documents = [
    "Campus Cafe is open from 8 AM to 8 PM on weekdays.",
    "The college library is open from 9 AM to 6 PM.",
    "Students can borrow up to 3 books for 14 days.",
    "Wi-Fi password for students is Campus@2026.",
    "The exam fee must be paid before the 10th of every semester month.",
    "Bus route 12 stops at the main gate every 30 minutes.",
]

print("Knowledge base ready. Total notes:", len(documents))
print()
for i, doc in enumerate(documents, start=1):
    print(f"  Note {i}: {doc}")


# %% CELL 3: Step 3 – Embedding (text → numbers)

sample_text = "The college library is open from 9 AM to 6 PM."

result = embed(model=EMBED_MODEL, input=sample_text)
vector = result.embeddings[0]

print("Original text:")
print(" ", sample_text)
print()
print("Embedding = list of numbers")
print("  Length:", len(vector))
print("  First 5 numbers:", vector[:5])
print()
print("We do not read these numbers by eye.")
print("We only use them to compare similarity.")


# %% CELL 4: Step 3b – Helper: get embedding for any text

def get_embedding(text):
    result = embed(model=EMBED_MODEL, input=text)
    return result.embeddings[0]


print("get_embedding() is ready.")


# %% CELL 5: Step 4 – Similarity score

def similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (n1 * n2)


question = "What are the library timings?"

note_related = "The college library is open from 9 AM to 6 PM."
note_unrelated = "Bus route 12 stops at the main gate every 30 minutes."

q_vec = get_embedding(question)
related_score = similarity(q_vec, get_embedding(note_related))
unrelated_score = similarity(q_vec, get_embedding(note_unrelated))

print("Question:", question)
print()
print(f"Related note score   : {related_score:.2f}  → {note_related}")
print(f"Unrelated note score : {unrelated_score:.2f}  → {note_unrelated}")
print()
print("Higher score = better match. This is how RETRIEVE works.")


# %% CELL 6: Step 4b – Retrieve top matching notes

def retrieve(question, documents, top_k=2):
    q_vec = get_embedding(question)

    scores = []
    for doc in documents:
        score = similarity(q_vec, get_embedding(doc))
        scores.append((score, doc))

    scores.sort(reverse=True)
    return scores[:top_k]


question = "What are the library timings?"
matches = retrieve(question, documents, top_k=2)

print("Question:", question)
print()
print("Top matching notes:")
for score, doc in matches:
    print(f"  score {score:.2f} | {doc}")


# %% CELL 7: Step 5 – Augment (build the prompt with notes)

question = "What are the library timings?"
matches = retrieve(question, documents, top_k=2)

context = "\n".join(doc for score, doc in matches)

prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

print("===== Prompt sent to LLM =====")
print(prompt)
print("================================")


# %% CELL 8: Step 6 – Generate (ask the LLM)

response = chat(
    model=CHAT_MODEL,
    messages=[{"role": "user", "content": prompt}],
)

print("Question:", question)
print()
print("Notes used (context):")
print(context)
print()
print("Final answer:", response.message.content)


# %% CELL 9: Step 7 – Full RAG in one place

def rag_answer(question, documents):
    # 1) RETRIEVE
    matches = retrieve(question, documents, top_k=2)
    context = "\n".join(doc for score, doc in matches)

    # 2) AUGMENT
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # 3) GENERATE
    response = chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return context, response.message.content


q1 = "How many books can I borrow?"
context1, answer1 = rag_answer(q1, documents)

print("TEST 1 – answer is in notes")
print("Question:", q1)
print("Context:\n" + context1)
print("Answer:", answer1)


# %% CELL 10: Step 7b – Question NOT in notes

q2 = "Who is the college principal?"
context2, answer2 = rag_answer(q2, documents)

print("TEST 2 – answer is NOT in notes")
print("Question:", q2)
print("Context:\n" + context2)
print("Answer:", answer2)
print()
print("If notes do not contain the answer → bot should say I don't know.")
