
import streamlit as st

def app():
    # Welcome Message
    st.title("Welcome to Omnia RAG Solutions")
    st.subheader("Empowering Your Business with LLM")

    # Visual Explanation of RAG (Retrieval-Augmented Generation)
    st.markdown("""
    ### What is RAG and How Does It Benefit Your Business?
    RAG (Retrieval-Augmented Generation) combines intelligent information retrieval with language models to deliver accurate, data-driven responses.
    """)

    # Step 1: Information Retrieval
    st.markdown("#### 🔍 Retrieval")
    st.write("RAG first searches your internal data sources to find relevant information.")

    # Step 2: Generation
    st.markdown("#### 💡 Generation")
    st.write("Using this information, a language model generates accurate and personalized responses to your needs.")

    # Step 3: Continuous Improvement
    st.markdown("#### 🔄 Continuous Improvement")
    st.write("With each interaction, the system optimizes itself to deliver more precise and effective results.")


    # Final Explanation on Authentication
    st.write("To access all features, create an account or log in using your Google account or email.")

    # Registration Redirection
    st.markdown("""
    ### 📝 Create an Account
    Navigate to the **Account** page to sign up and start your journey with RAG.
    """)
    
    # Note: 
    st.info("If you wish to contact me, feel free to reach out via LinkedIn. For more topics related to data science, visit my YouTube channel https://www.youtube.com/@Data_Pathfinder or my website at https://josh1313.github.io/Joshweb.io/")

    # Security Information
    st.markdown("""
    ### Security Notice
    For external access, you can configure a tunnel using ngrok. Simply create an account on ngrok, generate an authentication token, and add it to your .env file. This will create a secure tunnel for your localhost, ensuring it is not vulnerable.

    The app does not store cookies; it only saves your email for operational purposes. For transparency, all saved configurations are displayed, so you can review what is being stored. 

    The application has a Apache License http://www.apache.org/licenses/.
    """)