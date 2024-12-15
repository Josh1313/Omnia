import os, tempfile, uuid
import streamlit as st
import pandas as pd
from typing import List
import streamlit as st


from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from Utils.debug_helper import write_debug_log
from Utils.chroma_db_settings import Chroma
from Utils.chromadb_settings_openai import ChromaOpenai
from Utils.constants import CHROMA_SETTINGS, CHROMA_OPENAI_SETTINGS


# Load Variables globales
source_directory = os.environ.get('SOURCE_DIRECTORY', 'documents')

# Ensure embeddings are available
embeddings = st.session_state.get("embeddings")

# Verificar y cargar embeddings desde la sesión
if not st.session_state.get("embeddings"):
    st.error("No embeddings configured. Please configure a model first.")
    embeddings = None
else:
    embeddings = st.session_state["embeddings"]
    st.write(f"Using embeddings: {type(embeddings).__name__}")

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}

# Centralized function to get database settings
def get_db_settings_and_collection_name():
    embeddings = st.session_state.get("embeddings")
    if not embeddings:
        raise ValueError("Embeddings are not configured in session state.")
    
    # Determine configuration based on embeddings
    is_openai = "OpenAIEmbeddings" in str(type(embeddings))
    db_settings = CHROMA_OPENAI_SETTINGS if is_openai else CHROMA_SETTINGS
    collection_name = "vectordb_openai" if is_openai else "vectordb"
    chunk_size = 1000 if is_openai else 500
    chunk_overlap = 200 if is_openai else 50

    return db_settings, collection_name, chunk_size, chunk_overlap

def load_single_document(uploaded_file) -> List[Document]:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in LOADER_MAPPING:
        # Generar un nombre único para el archivo temporal
        tmp_filename = f"{uploaded_file.name}"
        tmp_path = os.path.join(tempfile.gettempdir(), tmp_filename)

        # Guardar temporalmente el archivo cargado con el nombre único
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
        
        try:
            # Crear una instancia del cargador correspondiente
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(tmp_path, **loader_args)
            return loader.load()
        finally:
            # Eliminar el archivo temporal después de usarlo
            os.unlink(tmp_path)

    raise ValueError(f"Unsupported file extension '{ext}'")


def get_unique_sources_df(db_settings, collection_name):
    """
    Returns a DataFrame of unique source file names from the specified ChromaDB settings and collection.
    """
    db_settings, collection_name, chunk_size, chunk_overlap = get_db_settings_and_collection_name()
    write_debug_log(f"Current use_openai flag: {collection_name}")

    collection = db_settings.get_collection(collection_name)
    df = pd.DataFrame(collection.get(include=['embeddings', 'documents', 'metadatas']))
    write_debug_log(f"df in document get unique .ingestion.py {df}")
    write_debug_log(f"document ingestion collections: {db_settings} y collections name {collection_name} ")

    # Suponiendo que 'df' es tu DataFrame original
    sources = df['metadatas'].apply(lambda x: x.get('source', None)).dropna().unique()

    # Obtener solo el nombre de archivo de cada ruta
    file_names = [source.split('/')[-1] for source in sources]

    # Crear un DataFrame con los nombres de archivo únicos
    unique_sources_df = pd.DataFrame(file_names, columns=['source'])

    # Mostrar el DataFrame con los diferentes valores de 'source'
    return unique_sources_df


# Modificar process_file para recibir el archivo cargado y el nombre
def process_file(uploaded_file, file_name):
    db_settings, collection_name, chunk_size, chunk_overlap = get_db_settings_and_collection_name()
    files_in_vectordb = get_unique_sources_df(db_settings, collection_name)['source'].tolist()
    write_debug_log(f"Current proces file use_openai flag: {collection_name}")

    if file_name in files_in_vectordb:
        return None
    else:
        # Convertir los bytes a documentos de texto
        documents = load_single_document(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts


def does_vectorstore_exist(db_settings,  collection_name):
    """
    Checks if vectorstore exists revisar el settings despues
    """
    db_settings, collection_name, _, _= get_db_settings_and_collection_name()
    collection = db_settings.get_or_create_collection(collection_name)
    write_debug_log(f"Current does vector funtion use_openai flag: {collection_name}")

    return collection


def ingest_file(uploaded_file, file_name):
    """
    Ingests a file into the appropriate vectorstore.
    """
    db_settings, collection_name, _, _ = get_db_settings_and_collection_name()
    write_debug_log(f"Current aca use_openai flag: {collection_name}")
    db_class = ChromaOpenai if "OpenAIEmbeddings" in str(type(st.session_state["embeddings"])) else Chroma
    
    db = db_class(embedding_function=st.session_state["embeddings"], client=db_settings, collection_name=collection_name)
    if does_vectorstore_exist(db_settings, collection_name):
        texts = process_file(uploaded_file, file_name)
        write_debug_log(f"Current does vector inside ingest file use_openai flag: {collection_name}")

        if texts == None:
            st.warning('This file has already been added previously.')
        else:
            st.spinner(f"Creando embeddings.")
            db.add_documents(texts)
            st.success(f"The file was successfully added.")
    else:
        # Create and store locally vectorstore
        st.success("Creating new vectorstore")
        texts = process_file(uploaded_file, file_name)
        st.spinner(f"Creating embeddings. May take some minutes...")
        db = db_class.from_documents(texts, embeddings, client=db_settings)
        st.success(f"The file was successfully added.")


def delete_file_from_vectordb(filename:str):
    """
    Deletes a file from the appropriate vectorstore.
    """
    db_settings, collection_name, _, _ = get_db_settings_and_collection_name()
    write_debug_log(f"Current use_openai delete funtion flag: {collection_name}")
    new_filename = '/tmp/' + filename
    try:
        current_collection = db_settings.get_collection(collection_name)
        current_collection.delete(where={"source": new_filename})
        print(f'The file was successfully: {filename} deleted')
    except:
        print(f'An error occurred while deleting the file. {filename}')
