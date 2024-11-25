import lancedb
import os
import re
import itertools
import threading
import time
import voyageai
from lancedb.pydantic import LanceModel, Vector
from config import VOYAGE_API_KEY, OPENAI_API_KEY
from lancedb.embeddings import get_registry
from openai import OpenAI
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

func = get_registry().get("openai").create(
  name="voyage-code-2",
  base_url="https://api.voyageai.com/v1/",
  api_key=VOYAGE_API_KEY,
)

class CodeChunks(LanceModel):
  filename: str
  text: str = func.SourceField()
  # 1536 is the embedding dimension of the `voyage-code-2` model.
  vector: Vector(1536) = func.VectorField()

db = lancedb.connect("/tmp/db")
table = db.open_table("code_chunks")
client = OpenAI(api_key=OPENAI_API_KEY)
code_block_pattern = re.compile(r'```(.+)\n([\S\s]*?)```')

# This will automatically use the environment variable VOYAGE_API_KEY.
# Alternatively, you can use vo = voyageai.Client(api_key="<your secret key>")
vo = voyageai.Client()

def spinner():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if not spinning:
            break
        print(f'\rLoading {c}', end='', flush=True)
        time.sleep(0.1)
    print('\r', end='', flush=True)

try:
  while True:
    query = input("> ")
    if query.lower() == 'exit':
      break
    
    # Start the spinner in a separate thread
    spinning = True
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.start()

    search_result = table.search(query).limit(50).to_pydantic(CodeChunks)
    documents = []
    for result in search_result:
      documents.append(result.filename + "\n\n" + result.text)

    reranking = vo.rerank(query, documents, model="rerank-2", top_k=5)
    documents_for_prompt = ""

    for result in reranking.results:
      documents_for_prompt += result.document + "\n\n\n\n"

    system_content = f"You are an expert programming assistant. Here are relevant code files {documents_for_prompt}"

    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": system_content},
          {"role": "user", "content": query},
      ],
    )
    # Stop the spinner
    spinning = False
    spinner_thread.join()

    response_text = response.choices[0].message.content
    matches = code_block_pattern.finditer(response_text)
    start = 0

    # Highlight code blocks
    for match in matches:
      print(response_text[start:match.start()])
      language = match.group(1) or "text"
      code = match.group(2)
      try:
        lexer = get_lexer_by_name(language, stripall=True)
      except Exception:
        lexer = get_lexer_by_name("text", stripall=True)
      highlighted_code = highlight(code, lexer, TerminalFormatter())
      print(highlighted_code)
      start = match.end()
    print(response_text[start:])
except KeyboardInterrupt:
  print("\nExiting...")
