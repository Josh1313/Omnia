import streamlit as st
import firebase_admin
from firebase_admin import credentials
import json
import requests

# Inicialización de Firebase Admin SDK
cred = credentials.Certificate("assistant-ai-70358-72c5969f02b8.json")
firebase_admin.initialize_app(cred)

# URLs de API de Firebase
API_KEY = "AIzaSyCublkcDhSsJQ4osuTdYH9wWQGg7VAutSw"
SIGN_UP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
SIGN_IN_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
RESET_PASSWORD_URL = "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"

# Ejecución principal de la app
def app():
    st.title('Welcome to :violet[Omnia] :robot_face:')

    # Configuración inicial de variables de sesión
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False  # Para controlar el estado de autenticación
    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'useremail' not in st.session_state:
        st.session_state['useremail'] = ""


    # Si el usuario está autenticado, mostrar mensaje de bienvenida y botón de cerrar sesión
    if st.session_state['authenticated']:
        st.write(f"Bienvenido, {st.session_state['username']} :wave:")

        # Botón de cerrar sesión
        if st.button('Cerrar sesión'):
            # Restablecer variables de sesión al estado inicial
            st.session_state['authenticated'] = False
            st.session_state['username'] = ""
            st.session_state['useremail'] = ""
            st.write("Has cerrado sesión correctamente. ¡No olvides registrarte la próxima vez!")

    else:
        # Mostrar formulario de login o signup si el usuario no está autenticado
        choice = st.selectbox('Login/Signup', ['Login', 'Sign up'])
        # Campos de entrada de email y password
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')

        # Lógica para Signup
        if choice == 'Sign up':
            username = st.text_input("Enter your unique username")
            if st.button('Create my account'):
                user = sign_up_with_email_and_password(email=email, password=password, username=username)
                if user:
                    # Configurar el estado de sesión para mostrar solo mensaje de bienvenida y botón de cerrar sesión
                    st.session_state['username'] = username
                    st.session_state['useremail'] = email
                    st.session_state['authenticated'] = True
                    st.balloons()
                    st.success("¡Cuenta creada con éxito! Disfruta de tu experiencia.")

        # Lógica para Login
        elif choice == 'Login':
            if st.button('Login'):
                userinfo = sign_in_with_email_and_password(email, password)
                if userinfo:
                    # Configurar el estado de sesión para mostrar solo mensaje de bienvenida y botón de cerrar sesión
                    st.session_state['username'] = userinfo['username']
                    st.session_state['useremail'] = userinfo['email']
                    st.session_state['authenticated'] = True
                    st.balloons()
                    st.success("Inicio de sesión exitoso. ¡Disfruta de tu experiencia!")
# Función de registro
def sign_up_with_email_and_password(email, password, username=None, return_secure_token=True):
    try:
        st.write("Intentando registrar usuario con correo:", email)
        
        # Preparar payload para solicitud de registro
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": return_secure_token
        }
        if username:
            payload["displayName"] = username 
        payload = json.dumps(payload)

        # Llamada a la API de registro
        r = requests.post(SIGN_UP_URL, params={"key": API_KEY}, data=payload)
        
        # Verificación de la respuesta
        # st.write("Respuesta de Firebase al intento de registro:", r.json())
        # Procesar la respuesta
        response_json = r.json()
        
        if "error" in  response_json:
            st.warning("Error en el registro: " +  response_json['error']['message'])
            return None
        else:
            st.success("Usuario registrado con éxito")
            return response_json.get('email')  # Devolver correo registrado si tiene éxito
    except Exception as e:
        st.warning(f'Error en el registro: {e}')

# Función de inicio de sesión
def sign_in_with_email_and_password(email=None, password=None, return_secure_token=True):
    try:
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": return_secure_token
        }
        payload = json.dumps(payload)
        r = requests.post(SIGN_IN_URL, params={"key": API_KEY}, data=payload)
        data = r.json()

        # Manejar error si existe en la respuesta
        if "error" in data:
            st.warning("Error en el inicio de sesión: " + data["error"]["message"])
            return None
        else:
            username = data.get('displayName', None)
            return {'email': data.get('email'), 'username': username}
    except Exception as e:
        st.warning(f'Error en el inicio de sesión: {e}')
        
            



