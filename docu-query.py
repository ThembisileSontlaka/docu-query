import os, re
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

class HandleRetrival:

    def __init__(self) -> None:
        self.embedding = self.get_embeddings()


    TEMPLATE = """
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""

# Step 1: Load Documents
    def load_documents(self, file_output_str):
        loaders = self.get_loaders(file_output_str)

        fileData = []
        for loader in loaders:
            fileData.extend(loader.load())

        embeddings = self.get_embeddings()

        return fileData


    def get_loaders(self, output_str):
        # Define a regular expression pattern to match the paths in brackets
        pattern = r'\((.*?)\)'

        # Use re.findall to extract all matches of the pattern
        paths = re.findall(pattern, output_str)

        loaders = []
        for path in paths :
            loaders.append(PyPDFLoader(path))

        return loaders

    # Step 2: Create embeddings
    def get_embeddings(self):
        api_token =  os.getenv("HUGGINGFACEHUB_API_TOKEN")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings':False}
        embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings


    def database_instance(self, pdfData, embeddings):
        db = DeepLake.from_documents(pdfData, embeddings)
        # question = "What is the aesthetic of horror?"
        # searchDocs = db.similarity_search(question)
        # print(searchDocs[0].page_content)
        return db


    # #Step 3: Retrieve
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

        #Step 4: Generate
        result = qa_chain({ "query" : "What is the aesthetic of horror?" })
        print(result["result"])