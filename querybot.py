import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DeepLake
from langchain.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

loaders = [
        PyPDFLoader("/home/thembisile/Documents/test-pdfs/animalFacts.pdf"),
        PyPDFLoader("/home/thembisile/Documents/test-pdfs/critical-self-reflection.pdf"),
        PyPDFLoader("/home/thembisile/Documents/test-pdfs/seasonsoftheyear.pdf"),
        ]

pdfData = []
for loader in loaders:
    pdfData.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splitData = text_splitter.split_documents(pdfData)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("API_TOKEN")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}

embeddings = HuggingFaceEmbeddings(
  model_name = model_name,
  model_kwargs = model_kwargs,
  encode_kwargs=encode_kwargs
)

db = DeepLake.from_documents(pdfData, embeddings)
# question = "Give me a fact about rats."
# searchDocs = db.similarity_search(question)
# print(searchDocs[0].page_content)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0, "max_length": 512},
)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say "I don't know", don't try to make up an answer. Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff", #Stuff all related data into the prompt as context to pass to the language model.
  retriever= db.as_retriever(),
  chain_type_kwargs={"prompt": PROMPT}
)
result = qa_chain({ "query" : "Give me a fact about rats." })
print(result["result"])

