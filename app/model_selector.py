
import os
import openai
import streamlit as st
from openai import OpenAI, BadRequestError
from subprocess import run, CalledProcessError, PIPE
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.embeddings import Embeddings
from Utils.streamlit_style import hide_streamlit_style
from typing import Optional, Tuple, Any



def app():
    hide_streamlit_style()
    
    if not st.session_state.get('authenticated', False):
        st.warning("Please log in to access Model Selector.")
        return
    
    st.title("Model Selector")

    # Leer lista de modelos dinámicamente desde variables de entorno
    DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
    model_list = os.getenv("MODEL_LIST", "llama3.2, gemma2:2b, tinyllama,  llama3,mistral,phi3,llama2,brxce/stable-diffusion-prompt-generator").split(",")
    embedding_model_list = os.getenv("EMBEDDING_MODEL_LIST", "all-MiniLM-L6-v2").split(",")
    ollama_host_default = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    
    def is_model_selected(model_name: Optional[str]) -> bool:
        """
        Verifica si el modelo ha sido seleccionado por el usuario.
        """
        if not model_name:
            st.warning("No model selected. Please select a model before proceeding.")
            return False
        return True

    def is_embedding_selected(embedding_name: Optional[str]) -> bool:
        """
        Verifica si un modelo de embedding ha sido seleccionado.
        """
        if not embedding_name:
            st.warning("No embedding selected. Please select an embedding before proceeding.")
            return False
        return True

    
    def validate_openai_api_key(api_key: str) -> bool:
        """
        Valida si una clave de API de OpenAI es correcta.
        Realiza una llamada de prueba a un endpoint válido (e.g., completions).
        """
        try:
            # Inicializar el cliente de OpenAI
            client = OpenAI(api_key=api_key)
            
            # Llamada de prueba al modelo más pequeño y rápido
            client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=5
        )
            return True  # Clave válida
        except BadRequestError as ai_err:
            # Extraer mensaje del error de OpenAI
            ai_response_msg = ai_err.body.get("message", "Unknown error")
            st.warning(f"Invalid OpenAI API Key: {ai_response_msg}. Please check your key and try again.")
            return False
        except Exception as e:
            # Manejar errores genéricos
            st.error(f"An unexpected error occurred: {str(e)}")
            return False
    
    # Validate Grok API Key
    def validate_grok_api_key(api_key: str, base_url: str) -> bool:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": "Hello Grok"}],
                max_tokens=5
            )
            return True
        except BadRequestError as e:
            st.warning(f"Invalid Grok API Key: {e.body.get('message', 'Unknown error')}")
            return False
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return False
    
    def get_container_id(image_name: str, container_name: str) -> Optional[str]:
        """
        Retrieves the container ID of a running container by image name or container name.
        """
        try:
            # Comando para obtener el ID del contenedor
            command = (
                f"docker ps --filter ancestor={image_name} --filter name={container_name} --format \"{{{{.ID}}}}\""
            )
            st.info(f"Executing command: {command}")  # Mensaje informativo en la UI

            # Ejecutar el comando
            result = run(command, shell=True, capture_output=True, text=True, check=True)

            # Procesar la salida
            container_id = result.stdout.strip()
            if container_id:
                st.success(f"Container ID found: {container_id}")
                return container_id
            else:
                st.warning(f"No running container matches: ancestor={image_name}, name={container_name}")
                return None

        except FileNotFoundError:
            st.error("Docker CLI not found. Ensure Docker is installed and in the PATH.")
            return None
        except CalledProcessError as e:
            st.error(f"Failed to execute Docker command. Details: {e.stderr.strip()}")
            return None
        
    def is_model_available(model_name: str, container_id: str) -> bool:
        """
        Verifica si un modelo ya está descargado en el contenedor de Ollama.
        """
        try:
            check_command = f"docker exec {container_id} ollama list"
            result = run(check_command.split(), capture_output=True, text=True, check=True)
            available_models = result.stdout.lower()
            return model_name.lower() in available_models
        except CalledProcessError as e:
            st.error(f"Error checking model availability: {e.stderr.strip()}")
            return False    

    def ensure_model_downloaded(model_name: str, image_name: str = "ollama/ollama:latest", container_name: str = "omnia-ollama-1", max_retries: int = 3) -> bool:
        """
        Ensures the specified model is downloaded in the running 'ollama' container.
        """
        st.info(f"Checking model availability: {model_name}...")
        
        # Get the container ID dynamically
        container_id = get_container_id(image_name, container_name)
        if not container_id:
            st.error("Could not find the required container to pull the model.")
            return False
        
        if is_model_available(model_name, container_id):
            st.success(f"Model '{model_name}' is already available!")
            return True
        
        # Si el modelo no está disponible, intenta descargarlo
        st.info(f"Downloading model '{model_name}'...")
        pull_command = f"docker exec {container_id} ollama pull {model_name}"
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                st.info(f"Attempt {retry_count + 1} to download model '{model_name}'...")
                result = run(pull_command.split(), capture_output=True, text=True, check=True)
                st.success(f"Model {model_name}  downloaded successfully!")
                return True
            
            except CalledProcessError as e:
                retry_count += 1
                st.error(f"Attempt {retry_count} failed. Error: {e.stderr.strip()}")
                if retry_count < max_retries:
                    st.info(f"Retrying in 5 seconds...")
                    import time
                    time.sleep(5)  

        st.error(f"Failed to download model '{model_name}' after {max_retries} attempts.")
        return False
    
    
    def configure_model() -> Tuple[Optional[Any], Optional[Any], Optional[str], Optional[str], Optional[str], Optional[bool]]:
        """
        Configura modelos y embeddings, permitiendo la selección entre OpenAI y modelos locales.
        """
        
        st.sidebar.subheader("Model Configuration")
        model_type = st.sidebar.radio("Select Model Type:", options=["Local (Custom Models)", "OpenAI","Grok"])
        
        llm, embeddings, openai_api_key,xai_api_key, base_url = None, None, None, None, None
        use_openai = False

        if model_type == "OpenAI":
            # OpenAI Configuration
            openai_api_key = st.sidebar.text_input("OpenAI API Key:", type="password", placeholder="Enter OpenAI API Key")
            if not openai_api_key:
                st.sidebar.warning("OpenAI API Key is required to use OpenAI models.")
            else:
                # Validar la clave API antes de proceder
                if not validate_openai_api_key(openai_api_key):
                    return None, None, None, None, None, False  # Detener la configuración si la clave es inválida
            
            llm_model = st.sidebar.selectbox("Choose OpenAI LLM Model:", options=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"], index=0)
            embedding_model = st.sidebar.selectbox("Choose OpenAI Embedding Model:", options=["text-embedding-ada-002"], index=0)

            if openai_api_key:
                llm = ChatOpenAI(model=llm_model, openai_api_key=openai_api_key)
                embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_api_key)
                use_openai = True
        
        elif model_type == "Grok":
            # Grok Configuration (new addition)
            xai_api_key  = st.sidebar.text_input("Grok API Key:", type="password", placeholder="Enter XAI API Key")
            base_url = st.sidebar.text_input("Grok Base URL:", value=DEFAULT_GROK_BASE_URL)
            if xai_api_key and validate_grok_api_key(xai_api_key, base_url):
                grok_model = st.sidebar.selectbox("Choose Grok Model:", options=["grok-beta","grok-2-vision-1212"])
                
                # Configurar cliente de Grok
                client = OpenAI(api_key=xai_api_key, base_url=base_url)
                llm = {"model_name": grok_model, "client": client}  # Almacenar modelo y cliente
                embeddings = "None"  # No necesitas embeddings para Grok en este ejemplo
            else:
                st.sidebar.warning("Valid Grok API Key is required.")

                
        else:
            # Selección dinámica de modelos locales
            local_model_name = st.sidebar.selectbox("Choose Local Model:", options=model_list, index=None)
            if not is_model_selected(local_model_name):
                return None, None, None, None, None, False  # Detener si no hay modelo seleccionado

            ollama_host = st.sidebar.text_input("Enter Ollama Host:", value=ollama_host_default, help="Default: http://localhost:11434")
            # Ensure the model is downloaded
            if ensure_model_downloaded(local_model_name):
                llm = {"model_name": local_model_name, "ollama_host": ollama_host}
            else:
                st.error(f"Could not configure the model '{local_model_name}'. Please try again.")
                return None, None, None, None, None,  False  # Abort configuration
            
            embedding_model_name = st.sidebar.selectbox("Choose Embedding Model:",options=embedding_model_list,index=None)
            if not is_embedding_selected(embedding_model_name):
                return None, None, None, None, None, False  # Detener si no hay embedding seleccionado
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        return llm, embeddings, openai_api_key, xai_api_key, base_url,  use_openai

    llm, embeddings, openai_api_key, xai_api_key, base_url, use_openai = configure_model()
    # Función para guardar la configuración en st.session_state
    if st.button("Guardar Configuración"):
        if llm and embeddings:
            st.session_state["llm"] = llm
            st.session_state["embeddings"] = embeddings or "None"  # Asegurar que exista un valor, incluso si es None
            st.session_state["openai_api_key"] = openai_api_key
            st.session_state["xai_api_key"] = xai_api_key  # Guardar XAI_API_KEY
            st.session_state["use_openai"] = use_openai
            if base_url:  # Guardar base_url solo si es Grok
                st.session_state["base_url"] = base_url
            if isinstance(llm, dict) and "ollama_host" in llm:
                st.session_state["ollama_host"] = llm["ollama_host"]
            st.success("Configuration saved successfully!")
        else:
            st.error("Please complete the model configuration before saving.")
    
    st.write("### Current Configuration:")
    if "llm" in st.session_state and st.session_state["llm"]:
        st.write(f"LLM Model: {st.session_state['llm']}")
    if "embeddings" in st.session_state and st.session_state["embeddings"]:
        st.write(f"Embeddings: {st.session_state['embeddings']}")
    if "use_openai" in st.session_state:
        st.write(f"Use OpenAI: {st.session_state['use_openai']}")    
    if "base_url" in st.session_state:
        st.write(f"Grok Base URL: {st.session_state['base_url']}")
    if "ollama_host" in st.session_state:
        st.write(f"Ollama Host: {st.session_state['ollama_host']}")


