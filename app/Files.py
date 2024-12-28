import streamlit as st
import chromadb, os
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from Utils.constants import CHROMA_SETTINGS, CHROMA_OPENAI_SETTINGS
from Utils.chromadb_settings_openai import get_unique_sources_df as get_unique_sources_openai
from Utils.chroma_db_settings import get_unique_sources_df  as get_unique_sources_local
from Utils.document_ingestion import ingest_file, delete_file_from_vectordb, get_db_settings_and_collection_name
from Utils.streamlit_style import hide_streamlit_style
from Utils.debug_helper import write_debug_log

def app():
    hide_streamlit_style()
    
    if not st.session_state.get('authenticated', False):
        st.warning("Please log in to access files.")
        return
    
    
    # Verificación de modelo configurado
    if not st.session_state.get("llm") or not st.session_state.get("embeddings"):
        st.warning("Please select a model before using Knowledge ..")
        st.info("Go to the Models tab to choose a model.")
        return  # Detenemos la ejecución si no se ha configurado un modelo
    
    # Cargar los embeddings desde la configuración
    embeddings = st.session_state["embeddings"]
    
    # Seleccionar el vector store dinámicamente
    use_openai = "OpenAIEmbeddings" in str(type(embeddings))
    
    # Obtener configuración del helper
    db_settings, collection_name, _, _ = get_db_settings_and_collection_name()
    
    get_unique_sources = get_unique_sources_openai if use_openai else get_unique_sources_local
    # Define the Chroma settings
    db_settings = CHROMA_OPENAI_SETTINGS if use_openai else CHROMA_SETTINGS
    

    st.title('Files')

    # Carpeta donde se guardarán los archivos en el contenedor del ingestor
    container_source_directory = 'documents'

    # Función para guardar el archivo cargado en la carpeta
    def save_uploaded_file(uploaded_file):
        # Verificar si la carpeta existe en el contenedor, si no, crearla
        if not os.path.exists(container_source_directory):
            os.makedirs(container_source_directory)

        with open(os.path.join(container_source_directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join(container_source_directory, uploaded_file.name)

    # Widget para cargar archivos
    uploaded_files = st.file_uploader("Load file", type=['csv', 'doc', 'docx', 'enex', 'eml', 'epub', 'html', 'md', 'odt', 'pdf', 'ppt', 'pptx', 'txt'], accept_multiple_files=False)

    # Botón para ejecutar el script de ingestión
    if st.button("Add File to  Database") and uploaded_files:
        file_name = uploaded_files.name
        ingest_file(uploaded_files, file_name)
    elif not uploaded_files:
        st.write("Please upload at least one file before adding it to the Database.")

    st.subheader('Files in Database:')

    try:
        # Obtener los archivos almacenados en la base de datos
        files = get_unique_sources(db_settings,collection_name)
        write_debug_log(f"Db settings in files.py : {db_settings} and your collection name :{collection_name}")
        write_debug_log(f"Files DataFrame:{files}")
        
        # Agregar columna de eliminación
        files['Delete'] = False
        files_df = st.data_editor(files, use_container_width=True)
        
        # Manejo de eliminación de archivos
        if len(files_df.loc[files_df['Delete']]) == 1:
            st.divider()
            st.subheader('Delete file')
            file_to_delete = files_df.loc[files_df['Delete'] == True]
            filename = file_to_delete.iloc[0, 0]
            st.write(filename)
            st.dataframe(file_to_delete, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
                        
            with col2:
                if st.button('Remove File from Database'):
                    try:
                        delete_file_from_vectordb(filename)
                        st.success('File successfully removed')
                        st.rerun()
                    except Exception as e:
                        st.error(f'An error occurred while deleting the file: {e}')
            
        elif len(files_df.loc[files_df['Delete']]) > 1:
            st.warning('Only one file can be deleted at a time.')
                        
    except Exception as e:
        st.error(f"An error occurred while fetching files: {e}")

