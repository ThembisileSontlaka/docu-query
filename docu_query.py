import os, re
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

class HandleRetrival:

    def __init__(self) -> None:
        self.db_path = self.get_db_path()
        huggingface_token =  os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.TEMPLATE = """
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
        {context}
        Question: {question}
        Helpful Answer:"""


    def get_db_path(self):
        db = "embeddings_db/"
        # Get the absolute path to the project directory
        project_dir = os.path.abspath(os.path.dirname(__file__))
        # Join the project directory with the relative path
        full_path = os.path.join(project_dir, db)
        return full_path


# Step 1: Load Documents
    def load_documents(self, file_output_str):
        loaders = self.get_loaders(file_output_str)

        fileData = []
        for loader in loaders:
            fileData.extend(loader.load())


        embeddings = self.get_embeddings()
        self.init_db(fileData, embeddings)
        return True


    def get_loaders(self, output_str):
        # Define a regular expression pattern to match the paths in brackets
        pattern = r'\((.*?)\)'

        # Use re.findall to extract all matches of the pattern
        paths = re.findall(pattern, output_str)

        loaders = []
        for path in paths:
            print(path)
            loaders.append(PyPDFLoader(path))

        return loaders

    # Step 2: Create embeddings
    def get_embeddings(self):
        print("Creating Embeddings")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings':False}
        embeddings = HuggingFaceEmbeddings(
            model_name = model_name,
            model_kwargs = model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings


    def init_db(self, pdfData, embeddings):
        print("Initiating Database")
        

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, 493, exist_ok=True)
            DeepLake.from_documents(pdfData, embedding=embeddings, overwrite = True)
            print("Database created!")
            return True

        return False

    # #Step 3: Retrieve
    def retriever(self, user_question):

        db = self.db_instance()
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0, "max_length": 512},
        )

        QA_PROMPT = PromptTemplate.from_template(self.TEMPLATE)
        
        qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT}
        )
        results = self.generate_response(qa_chain, user_question)

        return results


    def generate_response(self, qa_chain, user_question):
        #Step 4: Generate
        result = qa_chain({ "query" : user_question})
        return result
