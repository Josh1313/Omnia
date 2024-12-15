# Usa una imagen base de Python
FROM python:3.11

# Establecer el directorio de trabajo
WORKDIR /app

# Copia todo el contenido del directorio actual al directorio de trabajo
COPY . .

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt  

RUN pip install sentence-transformers torch  

# Verifica que sentence-transformers y torch estén correctamente instalados
RUN pip show sentence-transformers torch || (echo "Error: sentence-transformers o torch no instalado" && exit 1)

RUN python -m pip install --upgrade pip setuptools wheel

# Install Docker CLI
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*

# Verify Docker installation
RUN docker --version || (echo "Error: Docker CLI not installed" && exit 1)

# Expose the Streamlit port
EXPOSE 8080

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]

