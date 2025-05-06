from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPEN_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
parser = StrOutputParser()

template = """ Answer the question based on the context provided.

Context:
{context}

Question: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

file_path = './ref_doc.txt'
loader = TextLoader(file_path)
text_doc= loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=20)
text_splitter.split_documents(text_doc)[:10]
documents = text_splitter.split_documents(text_doc)
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorestore_xoprompt = DocArrayInMemorySearch.from_documents(
    documents,
    embeddings,
)

chain = (
    {"context": vectorestore_xoprompt.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
)
question = input("Enter a question: ")
chain.invoke({"question": question})

### Pinecone Implementation ------ ->> 
