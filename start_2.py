from requests import options

import ollama

response = ollama.list()
# print(response)

# chat example
res = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "hello how are you?"},
    ],
)
res_1 = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "user", "content": "hello how are you?"},
    ],
    stream=True,
)

ollama.create(
    model="knowitall",
    from_="llama3.2",
    system="you are very smart assistant who knows everything about Oceans",
)
res_3 = ollama.generate(model="knowitall", prompt="why is ocean salty")
print(res_3["response"])
# delete model
ollama.delete("knowitall")
