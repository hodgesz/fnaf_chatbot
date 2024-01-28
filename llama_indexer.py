import logging
import sys
import os
from pathlib import Path
import openai
from dotenv import load_dotenv, find_dotenv
import qdrant_client
from IPython.display import Markdown, display
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import TitleExtractor, SummaryExtractor
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import MetadataMode
from llama_index import download_loader


from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def create_index():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model_name = os.getenv('GPT3_5_MODEL_NAME')
    # model_name = os.getenv('GPT4_MODEL_NAME')
    # model_name = os.getenv('MIXTRAL_MODEL_NAME')
    temperature = os.getenv('MODEL_TEMPERATURE')
    embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
    qdrant_base_url = os.getenv('QDRANT_BASE_URL')
    qdrant_index = os.getenv('QDRANT_FNAF_INDEX')

    llm = OpenAI(model=model_name, temperature=temperature)
    embed_model = HuggingFaceEmbedding(model_name=embeddings_model_name)

    client = qdrant_client.QdrantClient(
        url=qdrant_base_url,
        prefer_grpc=True,
    )

    service_context = ServiceContext.from_defaults()
    vector_store = QdrantVectorStore(
        client=client, collection_name=qdrant_index, prefer_grpc=True
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=400, chunk_overlap=20),
            TitleExtractor(
                llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
            ),
            # SummaryExtractor(
            #      llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
            # ),
            embed_model,
        ],
        vector_store=vector_store
    )

    documents_path = "/Users/hodgesz/LangChain/data/fnaf/"

    loader = SimpleDirectoryReader(documents_path, recursive=False, exclude_hidden=True)

    # documents = loader.load_data(file=Path(documents_path))
    documents = loader.load_data()

    # service_context = ServiceContext.from_defaults(embed_model=embed_model)
    # index = VectorStoreIndex.from_vector_store(
    #     pipeline.vector_store, service_context=service_context
    # )

    nodes = pipeline.run(documents=documents)

    # index = VectorStoreIndex.from_documents(
    #     documents,
    #     storage_context=storage_context,
    #     service_context=service_context,
    #     use_async=True,
    # )
    #
    # # text_splitter = RecursiveCharacterTextSplitter(
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=400,
    #     chunk_overlap=20,
    #
    # )
    # docs_split = text_splitter.split_documents(docs)
    #
    # embeddings = OpenAIEmbeddings(
    #     model=os.environ["EMBEDDINGS_MODEL_NAME"],
    #     openai_api_key=os.environ["OPENAI_API_KEY"]
    # )
    #
    # pinecone.init(
    #     api_key=os.environ["PINECONE_API_KEY"],
    #     environment=os.environ["PINECONE_ENVIRONMENT"]
    # )
    #
    # Pinecone.from_documents(
    #     docs_split,
    #     embeddings,
    #     index_name=os.environ["PINECONE_INDEX"]
    # )


def main():
    _ = load_dotenv(find_dotenv())  # read local .env file

    create_index()


if __name__ == "__main__":
    main()
