# Ollama Python SDK – Simple Examples
# Copy each CELL into a Jupyter notebook cell and run it.
#
# Setup (once in terminal):
#   pip install ollama
#   ollama pull llama3.2


# %% CELL 0: Install
# %pip install ollama


# %% CELL 1: Connect and get a response

from ollama import generate

MODEL = "llama3.2"

response = generate(model=MODEL, prompt="What is an LLM? Answer in one sentence.")

print(response.response)


# %% CELL 2: Streaming response

from ollama import generate

MODEL = "llama3.2"

for chunk in generate(model=MODEL, prompt="Write a short poem about AI.", stream=True):
    print(chunk.response, end="", flush=True)


# %% CELL 3: Simple chat

from ollama import chat

MODEL = "llama3.2"

response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": "What is Python used for?"},
    ],
)

print(response.message.content)


# %% CELL 4: Chat with history

from ollama import chat

MODEL = "llama3.2"

messages = []

# Turn 1
messages.append({"role": "user", "content": "My name is Priya."})
response = chat(model=MODEL, messages=messages)
messages.append({"role": "assistant", "content": response.message.content})
print("Assistant:", response.message.content)

# Turn 2 — model remembers the name
messages.append({"role": "user", "content": "What is my name?"})
response = chat(model=MODEL, messages=messages)
messages.append({"role": "assistant", "content": response.message.content})
print("Assistant:", response.message.content)
