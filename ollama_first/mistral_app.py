import ollama

stream = ollama.chat(
    model='mistral',
    messages=[{'role': 'user', 'content': 'What is Flagger Force?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)