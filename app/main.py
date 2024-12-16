import streamlit as st

from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
        page_title="Assistant",
)

st.markdown(
    """
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src=f"https://www.googletagmanager.com/gtag/js?id={os.getenv('analytics_tag')}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', os.getenv('analytics_tag'));
        </script>
    """, unsafe_allow_html=True)

print(os.getenv('analytics_tag'))

import home as home, account as account, Chat as Chat, Files as Files, model_selector as model_selector






class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        with st.sidebar:        
            app = option_menu(
                menu_title='Omnia AI 🤖',
                options=['Home','Account','Chat','KnowledgeBase', 'Models'],
                icons=['house-fill','person-circle','chat-fill','file-earmark-text-fill','gear-fill'],
                menu_icon=" :robot_face: ",
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
        "icon": {"color": "white", "font-size": "23px"}, 
        "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == "Home":
            home.app()
        if app == "Account":
            account.app()    
        if app == 'Chat':
            Chat.app()  
        if app == 'KnowledgeBase':
            Files.app()
        if app == 'Models':
            model_selector.app()

    run()            

