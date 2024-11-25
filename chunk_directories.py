import os
import lancedb
from pathlib import Path
from transformers import GPT2Tokenizer
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from config import VOYAGE_API_KEY

# Initialize the tokenizer (replace with voyage-code-2 tokenizer)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Replace with your tokenizer
MAX_TOKENS = 16000  # Maximum context length for voyage-code-2

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

# Truncate file content to fit the context length
def truncate_file_content(content, max_tokens=MAX_TOKENS):
  tokens = tokenizer.encode(content, truncation=False)
  truncated_tokens = tokens[:max_tokens]  # Truncate tokens to the first 16,000
  return tokenizer.decode(truncated_tokens)  # Convert back to text

# Process the repository
def process_repository(repo_path):
  db_path="/tmp/db"
  db = lancedb.connect(db_path)
  table = db.create_table("code_chunks", schema=CodeChunks, mode="overwrite", on_bad_vectors="drop")
  """Processes a repository, truncates files if needed, and returns a list of CodeChunks."""
  chunks = []
  excluded_dirs = ['vendor', 'node_modules', 'build', 'dist']  # List of directories to exclude
  for filepath in Path(repo_path).rglob("*.rb"): # TODO: Add other project files
    if any(excluded_dir in str(filepath.parts) for excluded_dir in excluded_dirs):
      print(f"Skipping excluded path: {filepath}")
      continue
    print("Processing", filepath)
    if filepath.is_file() and filepath.suffix in [".rb"]:  # TODO: Add other project files
      with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        truncated_content = truncate_file_content(content)
        table.add([{"filename": str(filepath), "text": truncated_content}])
        print(f"Stored {filepath} file in the LanceDB table.")
  return chunks

# Run the script
repo_path = "/Users/filippohr/Documents/anywhere" # TODO: Move repo_path to env
process_repository(repo_path)
