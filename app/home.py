
import streamlit as st

def app():
    # Bienvenida
    st.title("Bienvenido a Omnia RAG Solutions")
    st.subheader("Empoderando a tu empresa con LLM")

    # Explicación visual de RAG (Retrieval-Augmented Generation)
    st.markdown("""
    ### ¿Qué es RAG y cómo beneficia a tu empresa?
    RAG (Generación con Recuperación Aumentada) combina la búsqueda inteligente de información con modelos de lenguaje para ofrecer respuestas precisas y basadas en datos.
    """)

    # Paso 1: Recuperación de Información
    st.markdown("#### 🔍 Recuperación")
    st.write("RAG primero busca en tus fuentes de datos internas para encontrar información relevante.")

    # Paso 2: Generación
    st.markdown("#### 💡 Generación")
    st.write("Con esa información, un modelo de lenguaje genera respuestas precisas y personalizadas para tus necesidades.")

    # Paso 3: Mejora Continua
    st.markdown("#### 🔄 Mejora Continua")
    st.write("Con cada interacción, el sistema se optimiza para ofrecer resultados más precisos y eficaces.")

    # Explicación final sobre autenticación
    st.write("Para acceder a todas las funcionalidades, crea una cuenta o inicia sesión con tu cuenta de Google o correo electrónico.")

    # Redirección al registro
    st.markdown("""
    ### 📝 Crear una cuenta
    Dirígete a la página de **Cuenta** para registrarte e iniciar tu viaje con RAG.
    """)

    # Nota: Eliminación de la funcionalidad de publicación
    st.info("La funcionalidad de publicación ha sido desactivada en esta versión. En su lugar, enfócate en explorar los beneficios de RAG para tu empresa.")
