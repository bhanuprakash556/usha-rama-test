# FAISS – Ingestion and Retrieval of Chunks
# Use case: Load campus_notes.txt → chunk → store in FAISS → search
#
# Files needed in same folder:
#   campus_notes.txt
#
# Setup (run once in terminal):
#   pip install ollama faiss-cpu numpy
#   ollama pull nomic-embed-text
#
# Copy each CELL into Jupyter and run one by one.


# %% CELL 0: Setup

from ollama import embed
import numpy as np
import faiss

EMBED_MODEL = "nomic-embed-text"

print("Setup done.")


# %% CELL 1: What are we building?

print("""
========== FAISS Flow ==========

  1. Load file  = read campus_notes.txt
  2. Chunks     = each line becomes one chunk
  3. Embeddings = each chunk becomes numbers
  4. Ingest     = store those numbers in FAISS
  5. Retrieve   = ask a question → get closest chunks

FAISS = fast library to search similar vectors
================================
""")


# %% CELL 2: Load document and create chunks
# File: campus_notes.txt  (one fact per line = one chunk)

DOC_PATH = "campus_notes.txt"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Split by lines and remove empty lines
chunks = [line.strip() for line in text.splitlines() if line.strip()]

print("Loaded file:", DOC_PATH)
print("Total chunks:", len(chunks))
print()
for i, chunk in enumerate(chunks):
    print(f"  chunk[{i}]: {chunk}")


# %% CELL 3: Helper – convert text to embedding (numbers)

def get_embedding(text):
    result = embed(model=EMBED_MODEL, input=text)
    return result.embeddings[0]


# Test with one chunk
sample = get_embedding(chunks[0])
print("One chunk embedding:")
print("  Length:", len(sample))
print("  First 5 numbers:", sample[:5])


# %% CELL 4: INGESTION – convert all chunks to embeddings

embeddings = []
for chunk in chunks:
    embeddings.append(get_embedding(chunk))

# FAISS needs a NumPy array with type float32
embedding_matrix = np.array(embeddings, dtype="float32")

print("Embedding matrix shape:", embedding_matrix.shape)
print("  rows    = number of chunks :", embedding_matrix.shape[0])
print("  columns = embedding size   :", embedding_matrix.shape[1])


# %% CELL 5: INGESTION – store embeddings in FAISS index

dimension = embedding_matrix.shape[1]

# IndexFlatL2 = simple exact search (best for beginners / small data)
index = faiss.IndexFlatL2(dimension)

# Add all chunk embeddings into FAISS
index.add(embedding_matrix)

print("FAISS index ready.")
print("Total vectors stored:", index.ntotal)


# %% CELL 6: RETRIEVAL – search FAISS with a question

question = "What are the library timings?"

# 1) Convert question to embedding
question_vec = np.array([get_embedding(question)], dtype="float32")

# 2) Search top 2 closest chunks
k = 2
distances, indices = index.search(question_vec, k)

print("Question:", question)
print()
print("Retrieved chunks:")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"  {rank}. distance={dist:.2f} | chunk[{idx}]: {chunks[idx]}")

print()
print("Note: smaller distance = more similar chunk")


# %% CELL 7: RETRIEVAL – try another question

question = "How many books can I take?"

question_vec = np.array([get_embedding(question)], dtype="float32")
distances, indices = index.search(question_vec, k=2)

print("Question:", question)
print()
print("Retrieved chunks:")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    print(f"  {rank}. distance={dist:.2f} | chunk[{idx}]: {chunks[idx]}")


# %% CELL 8: Full helper functions (ingest + retrieve)

def ingest(chunks):
    """Create FAISS index from a list of text chunks."""
    vectors = [get_embedding(chunk) for chunk in chunks]
    matrix = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    return index


def retrieve(question, index, chunks, k=2):
    """Return top-k chunks for a question."""
    q_vec = np.array([get_embedding(question)], dtype="float32")
    distances, indices = index.search(q_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((float(dist), chunks[idx]))
    return results


# Build index again using the helper
faiss_index = ingest(chunks)

# Search
question = "When should I pay the exam fee?"
results = retrieve(question, faiss_index, chunks, k=2)

print("Question:", question)
print()
for dist, chunk in results:
    print(f"  distance={dist:.2f} | {chunk}")


# %% CELL 9: Save and load FAISS index (optional)

# Save index to disk
faiss.write_index(faiss_index, "campus_faiss.index")
print("Saved: campus_faiss.index")

# Load index back
loaded_index = faiss.read_index("campus_faiss.index")
print("Loaded vectors:", loaded_index.ntotal)

# Search again with loaded index
question = "What is the Wi-Fi password?"
results = retrieve(question, loaded_index, chunks, k=1)

print()
print("Question:", question)
for dist, chunk in results:
    print(f"  distance={dist:.2f} | {chunk}")
