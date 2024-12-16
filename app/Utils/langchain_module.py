from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from Utils.chroma_db_settings import Chroma
from Utils.chromadb_settings_openai import ChromaOpenai
from Utils.assistant_prompt import assistant_prompt
import streamlit as st
import os
import argparse


# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',5))

from Utils.constants import CHROMA_SETTINGS, CHROMA_OPENAI_SETTINGS


def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


def response(query:str) -> str:
    # Parse the command line arguments
    args = parse_arguments()
    
    # Validar configuración dinámica
    if not st.session_state.get("embeddings") or not st.session_state.get("llm"):
        st.error("No embeddings or LLM configured. Please configure these settings in the Model Selector.")
        return "Configuration missing. Please select a model and embeddings."

    # Cargar embeddings dinámicos (si existen)
    embeddings = st.session_state["embeddings"]
    if embeddings:
        st.write(f"Using embeddings: {type(embeddings).__name__}")
        # Seleccionar la configuración correcta de Chroma settings
        use_openai = "OpenAIEmbeddings" in str(type(embeddings))
        db_settings = CHROMA_OPENAI_SETTINGS if use_openai else CHROMA_SETTINGS
        db_class = ChromaOpenai if use_openai else Chroma 
        db = db_class(client=db_settings, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    else:
        st.warning("No embeddings configured. Proceeding without retrieval functionality.")
        retriever = None  # Sin embeddings, el flujo se limitará al modelo LLM.
    
    # Determinar el modelo LLM configurado
    llm_config = st.session_state["llm"]
    if "ollama_host" in llm_config:
        # Configurar modelo local
        st.write(f"Using local model: {llm_config['model_name']} with host {llm_config['ollama_host']}")
        llm = Ollama(model=llm_config["model_name"],
                    callbacks=[] if args.mute_stream else [StreamingStdOutCallbackHandler()],
                    temperature=0,
                    base_url=llm_config["ollama_host"],
        )
    elif "client" in llm_config and "model_name" in llm_config:
        # Configurar modelo Grok
        model_name = llm_config["model_name"]
        client = llm_config["client"]
        st.write(f"Using Grok model: {model_name}")
        
        
        # Realizar la solicitud directamente al cliente de OpenAI (Grok)
        def grok_query(client, model_name, user_query):
            # Obtener el mensaje del assistant_prompt
            prompt = assistant_prompt().format_prompt(question=user_query, context="")
            messages = [
                {"role": "system", "content": prompt.messages[0].content},
                {"role": "user", "content": user_query},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=500,
            )
            return response.choices[0].message.content

        return grok_query(client, model_name, query)     
    else:
        # Configurar modelo OpenAI
        st.write(f"Using OpenAI model: {llm_config}")
        llm = llm_config  # Este ya es un objeto LLM configurado desde `model_selector.py`

    # Procesar el flujo completo si hay embeddings
    if retriever:
        prompt = assistant_prompt()
        

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)
    else:
        # Si no hay embeddings, directamente interactuar con el modelo LLM
        return llm.predict(query)

