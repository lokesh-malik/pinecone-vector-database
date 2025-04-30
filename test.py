# import Libraries

import openai
import langchain
import pinecone 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

import os
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
doc=read_doc('documents/')
len(doc)

## Divide the docs into chunks
### https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html#
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs
documents=chunk_data(docs=doc)
len(documents)

## Embedding Technique Of OPENAI
embeddings = OpenAIEmbeddings(api_key="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")
embeddings

vectors=embeddings.embed_query("How are you?")
len(vectors)

## Vector Search DB In Pinecone
pinecone.init(
    api_key="pcsk_8ffCz_8932ui7hFxrBL1FZEGkuTF8ReGX5edN5gmo69HP8pEE8wUTkYWSg9xyYjeF1A13",
    environment="us-east-1"
)
index_name="lokeshmalik"

index=Pinecone.from_documents(doc,embeddings,index_name=index_name)
## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")
## Search answers from VectorDB
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response
our_query = "How much the agriculture target will be increased by how many crore?"
answer = retrieve_answers(our_query)
print(answer)

