from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate


class Chatbot:
    def __init__(self):
        self.vector_store = Chroma(embedding_function=OllamaEmbeddings(
            model='nomic-embed-text'), persist_directory="db/rag_net_eng")
        self.store = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
            parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000)
        )
        self.llm = ChatOllama(model="mistral:instruct", temperature=0)
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "Você é um assistente pessoal e sua função é responder as perguntas do usuário usando informações relevantes presentes no contexto fornecido.\n"
                        "Sempre responda em português\n"
                        "É proibido responder as perguntas com conhecimento que não está presente no contexto fornecido.\n"
                        "IMPORTANTE: Responda exclusivamente usando o contexto fornecido.\n"
                        "Contexto: Este é um documento técnico sobre comunicação de dados e redes industriais. O documento contém informações sobre protocolos de comunicação, redes de computadores, comunicação de dados, protocolos de rede, protocolos de transporte, protocolos de aplicação\n"
                        "{context}"
                    )
                ),
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )
        self.chain = {
            "context": self.retriever,
            "question": RunnablePassthrough(),
        } | self.chat_template | self.llm | StrOutputParser()

    def chatbot_response(self, query):
        for chunks in self.chain.stream(query):
            yield chunks

    def ask(self, query: str):
        return self.chatbot_response(query)


if __name__ == "__main__":
    chat = Chatbot()
    for text in chat.ask("o que é o protocolo TCP/IP? De qual seção você tirou essa informação?"):
        print(text, end="", flush=True)
