
import streamlit as st
from core.chatbot import Chatbot

# inicializa a instância
chat = Chatbot()
st.set_page_config(page_title="Chatbot IndusNetCom", page_icon=":robot:")
st.title("💬 Chatbot IndusNetCom")

with st.sidebar:
    "Em desenvolvimento"

# Inicialize a lista de mensagens se ainda não estiver inicializada
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Configuração da página
st.caption(
    "🚀Um chatbot desenvolvido para estudo de comunicação de dados e redes industriais")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá, como posso ajudá-lo?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        aux = ""
        with st.chat_message("assistant"):
            with st.spinner("Pensando... "):
                msgs = chat.ask(prompt)
                container = st.empty()
                for msg in msgs:
                    aux += msg
                    container.markdown(aux)
            st.session_state.messages.append(
                {"role": "assistant", "content": aux})
