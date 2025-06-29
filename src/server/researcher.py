from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import os

class Researcher:
    def _get_vector_store(self):
        load_dotenv()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        return AstraDBVectorStore(
            collection_name="reasearch_agent",
            embedding=embeddings,
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace=os.getenv("ASTRA_DB_KEYSPACE"),
        )

    def add_urls(self, urls: List[str]):
        """Accepts a list of URLs and stores them."""
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)
        vector_store = self._get_vector_store()
        vector_store.add_documents(documents=docs)
        return {"status": "success", "stored_urls": urls}

    def query(self, question: str):
        vector_store = self._get_vector_store()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=256)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
        response = chain.invoke({"question": question})
        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"]
        }
