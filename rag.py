import os, ast
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DeepLake
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


TEMPLATE = """
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""


# Step 1: Load Documents
def load_documents(file_paths_string):
    loaders = get_loaders(file_paths_string)
   
    pdfData = []
    for loader in loaders:
        pdfData.extend(loader.load())

    return pdfData


def get_loaders(path_string):
    file_paths_list = ast.literal_eval(path_string)

    loaders = []
    for path in file_paths_list :
        loaders.append(PyPDFLoader(path))

    return loaders

# # Step 2: Split
def get_embeddings(pdfData):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splitData = text_splitter.split_documents(pdfData)

    # Step 3: Create embeddings
    # access_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings':False}
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings

def database_instance(pdfData, embeddings):
    db = DeepLake.from_documents(pdfData, embeddings)
    # question = "What is the aesthetic of horror?"
    # searchDocs = db.similarity_search(question)
    # print(searchDocs[0].page_content)
    return db

# #Step 4: Retrieve
def retrive_documents():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0, "max_length": 512},
    )

    return llm


def template():
    
    QA_PROMPT = PromptTemplate.from_template(template)
    # return QA_PROMPT
    
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": QA_PROMPT}
    )

    #Step 5: Generate
    result = qa_chain({ "query" : "What is the aesthetic of horror?" })
    print(result["result"])