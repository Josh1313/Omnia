import streamlit as st
from Utils.langchain_module import response
from Utils.streamlit_style import hide_streamlit_style

def app():
    hide_streamlit_style()
    
    # Verificación de autenticación
    if not st.session_state.get('authenticated', False):
        st.warning("Por favor, inicia sesión para acceder al chat.")
        return  # Detenemos la ejecución si el usuario no está autenticado
    
    # Verificación de modelo configurado
    if not st.session_state.get("llm") or not st.session_state.get("embeddings"):
        st.warning("Por favor, selecciona un modelo antes de usar el chat.")
        st.info("Ve a la pestaña de configuración para elegir un modelo.")
        return  # Detenemos la ejecución si no se ha configurado un modelo
    # Título de la aplicación Streamlit
    st.title("Omnia AI")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Manejo de entrada de usuario y respuesta del LLM
    if user_input := st.chat_input("write your messague here 🔥"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

    if user_input != None:
        if st.session_state.messages and user_input.strip() != "":
            assistant_response = response(user_input)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})