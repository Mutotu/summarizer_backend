import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# def load_document(file):
#   from langchain.document_loaders import PyPDFLoader
#   print(f'Loading {file}')
#   loader = PyPDFLoader(file)
#   data = loader.load()
#   return data

# data = load_document('./us_constitution.pdf')
# print(data[1])

# Funciton for file format
def load_document(file):
  import os
  name, extension = os.path.splitext(file)
  
  if extension == '.pdf':
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {file}')
    loader = PyPDFLoader(file)
  elif extension == '.docx':
    from langchain.document_loaders import Docx2txtLoader
    print(f'Loading {file}')
    loader = Docx2txtLoader(file)
    print(loader)
  else:
    print('Document format is not supported')
    return None
  data = loader.load()
  return data


# data = load_document('./the_great_gatsby.docx')
# print(data)

# Wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
  from langchain.document_loaders import WikipediaLoader
  loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
  data = loader.load()
  return data

data = load_from_wikipedia('GPT-4')
print(data[0].page_content)