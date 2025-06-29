from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.globals import set_debug
import os

load_dotenv()

set_debug(True)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4.1", temperature=0.9)

# breakpoint()
vector_store_explicit_embeddings = AstraDBVectorStore(
    collection_name="astra_vector_langchain",
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=os.getenv("ASTRA_DB_KEYSPACE"),
)
vector_store = vector_store_explicit_embeddings

documents_to_insert = [
    Document(
        page_content="ZYX, just another tool in the world, is actually my agent-based superhero",
        metadata={"source": "tweet"},
        id="entry_00",
    ),
    Document(
        page_content="I had chocolate chip pancakes and scrambled eggs "
        "for breakfast this morning.",
        metadata={"source": "tweet"},
        id="entry_01",
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and "
        "overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
        id="entry_02",
    ),
    Document(
        page_content="Building an exciting new project with LangChain "
        "- come check it out!",
        metadata={"source": "tweet"},
        id="entry_03",
    ),
    Document(
        page_content="Robbers broke into the city bank and stole "
        "$1 million in cash.",
        metadata={"source": "news"},
        id="entry_04",
    ),
    Document(
        page_content="Thanks to her sophisticated language skills, the agent "
        "managed to extract strategic information all right.",
        metadata={"source": "tweet"},
        id="entry_05",
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this "
        "review to find out.",
        metadata={"source": "website"},
        id="entry_06",
    ),
    Document(
        page_content="There are top 10 soccer players in the world right now.",
        metadata={"source": "website"},
        id="entry_07",
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, "
        "agentic applications!",
        metadata={"source": "tweet"},
        id="entry_08",
    ),
    Document(
        page_content="The stock market is down 500 points today due to "
        "fears of a recession.",
        metadata={"source": "news"},
        id="entry_09",
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
        id="entry_10",
    )
    # Document(
    #     page_content="ZYX, just another tool in the world, is actually my agent-based superhero",
    #     metadata={"source": "tweet"},
    #     id="entry_00",
    # ),
    # Document(
    #     page_content="I had chocolate chip pancakes and scrambled eggs "
    #     "for breakfast this morning.",
    #     metadata={"source": "tweet"},
    #     id="entry_01",
    # )
]
# vector_store.add_documents(documents=documents_to_insert)


# results = vector_store.similarity_search(
#     "Creating an innovative",
#     k=3,
# )

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
# breakpoint()
print(chain.invoke({"question": "how many top soccer players are there ?"}, return_intermediate_steps = True))


# print(f"Results: {result['answer']}")
# for result in results:
#     print(f"Content: {result.page_content}, Metadata: {result.metadata}, ID: {result.id}")