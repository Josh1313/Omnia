import os, chromadb
from chromadb.config import Settings

# Define the folder for storing database
#PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

# Define the Chroma settings
CHROMA_SETTINGS = chromadb.HttpClient(
    host="chroma1",#"host.docker.internal",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
# Configuración para Chroma con OpenAI
CHROMA_OPENAI_SETTINGS = chromadb.HttpClient(
    host="chroma2",#"host.docker.internal",
    port=8001,  # Puedes usar otro puerto si es necesario
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)
