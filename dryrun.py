from llama_index.core import SimpleDirectoryReader
docs = SimpleDirectoryReader("data").load_data()
print(f"I see {len(docs)} document objects.") 

from llama_index.core import SimpleDirectoryReader
reader = SimpleDirectoryReader("data")
# This shows exactly which parser is mapped to each file extension
print("Supported extensions:", reader.file_extractor.keys())