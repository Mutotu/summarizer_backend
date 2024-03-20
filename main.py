import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_document(file):
  from langchain.document_loaders import PyPDFLoader
  print(f'Loading {file}')
  loader = PyPDFLoader(file)
  data = loader.load()
  return data

data = load_document('./us_constitution.pdf')
print(data[1])