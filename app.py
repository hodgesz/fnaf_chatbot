import os

import chainlit as cl
import qdrant_client
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI, ChatAnyscale
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.vectorstores import Qdrant

load_dotenv()

embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=embeddings_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
qdrant_base_url = os.getenv('QDRANT_BASE_URL')
qdrant_index = os.getenv('QDRANT_FNAF_INDEX')
client = qdrant_client.QdrantClient(url=qdrant_base_url, prefer_grpc=True)
vector_store = Qdrant(client=client, collection_name=qdrant_index, embeddings=embeddings,
                      content_payload_key='_node_content')

# model_name = os.getenv('GPT3_5_MODEL_NAME')
# model_name = os.getenv('GPT4_MODEL_NAME')
model_name = os.getenv('MIXTRAL_MODEL_NAME')
temperature = os.getenv('MODEL_TEMPERATURE')

llm = ChatAnyscale(model_name=model_name,
                 temperature=temperature,
                 streaming=True)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4, 'score_threshold': 0.5})

template = """
You are a helpful chatbot with expert knowledge about the Five Nights at Freddy's conte having a conversation with a fan.

The answer and your response should only come from the Context section. If the question is not related to the Context section, politely respond that you are tuned to only answer questions that are related to Five Nights at Freddy's. Use as much detail as possible when responding.

Follow exactly these 3 steps:
1. Read the Context section below combining this information with the Question
2. Answer the Question using only the sources in the Context section
3. Your answer should use all applicable sources in the Context section
4. Only respond with URLs if you are sure they are relevant to the question, valid, and located in the Context section
5. Don't list or mention the words 'Sources', 'References', 'Context' or your confidence in your responses.

### CONTEXT
{context}

### QUESTION
{question}               
"""
prompt = ChatPromptTemplate.from_template(template)


@cl.on_chat_start
async def start_chat():
    rag_qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt
        }
        # max_tokens_limit=4096
    )

    cl.user_session.set("rag_qa_chain", rag_qa_chain)


@cl.on_message
async def main(message: cl.Message):
    rag_qa_chain = cl.user_session.get("rag_qa_chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True

    res = await rag_qa_chain.acall(message.content, callbacks=[cb])
    # res = await rag_qa_chain.acall(message.content, callbacks=[cb, tracer])
    # answer = res["answer"]
    # answer = res["result"]
    sources = ""
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            sources += f"\nSources: {', '.join(source_names)}"
        else:
            sources += "\nNo sources found"

    # await cl.Text(name="simple_text", content=answer, display="inline").send()
    await cl.Message(content=sources, elements=text_elements).send()
