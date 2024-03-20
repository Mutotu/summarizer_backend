import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


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


# Wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
  from langchain.document_loaders import WikipediaLoader
  loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
  data = loader.load()
  return data

# data = load_from_wikipedia('GPT-4')
# print(data[0].page_content)

def chunk_data(data, chunk_size=256):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
  chunks = text_splitter.split_documents(data)
  return chunks
  
  
chunks = chunk_data(load_document('./us_constitution.pdf'))
# print(chunks)


def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    


def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    
    pc = pinecone.Pinecone()
        
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

    # loading from existing index
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(
                environment='gcp-starter'
            )
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store
    
    
def delete_pinecone_index(index_name='all'):
  import pinecone
  pc = pinecone.Pinecone()
  
  if index_name == 'all':
    indexes = pc.list_indexes().names()
    print('Deleting all indexes ... ')
    for index in indexes:
        pc.delete_index(index)
    print('Ok')
  else:
    print(f'Deleting index {index_name} ...', end='')
    pc.delete_index(index_name)
    print('Ok')
    
delete_pinecone_index()    

vector_store = insert_or_fetch_embeddings('askadocument', chunks)