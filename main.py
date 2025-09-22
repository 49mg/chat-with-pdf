import os
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


class PdfChatbot:
    """
    A chatbot that can answer questions based on a provided PDF document.
    """

    def __init__(self, model_name: str = "gemma3:1b"):
        self.model = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.chain = None
        self.prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

        <context>
        {context}
        </context>

        Question: {input}""")

    def ingest(self, file_path: str):
        """
        Loads, splits, and indexes the PDF document to prepare for questioning.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' was not found.")

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()

        document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.chain = create_retrieval_chain(retriever, document_chain)
        print("PDF processed successfully.")

    def ask(self, question: str):
        """
        Asks a question to the chatbot and returns the answer.
        """
        if self.chain is None:
            return "Please upload a PDF document first to enable the chat."

        response = self.chain.invoke({"input": question})
        return response.get("answer", "Could not find an answer.")


chatbot = PdfChatbot()


def process_pdf(file):
    """
    Handles the PDF file upload, ingests it using the chatbot,
    and returns a confirmation message.
    """
    if file is not None:
        try:
            chatbot.ingest(file.name)
            return "PDF processed successfully! You can now ask questions.", gr.update(interactive=True,
                                                                                       placeholder="Ask a question about the PDF...")
        except Exception as e:
            return f"An error occurred: {e}", gr.update(interactive=False)
    return "No file uploaded.", gr.update(interactive=False)


def add_text(history, text):
    """
    Adds the user's question to the chat history in the correct format.
    """
    if not text or not text.strip():
        return history, gr.update(value=text)

    history.append({"role": "user", "content": text})
    return history, gr.update(value="", interactive=False)


def bot(history):
    """
    Generates a response and appends it to the history.
    """
    if not history or history[-1]['role'] != 'user':
        return history, gr.update(interactive=True)

    question = history[-1]['content']
    answer = chatbot.ask(question)
    history.append({"role": "assistant", "content": answer})

    return history, gr.update(interactive=True)


with gr.Blocks() as demo:
    gr.Markdown("# Chat with Your PDF")
    gr.Markdown("Upload a PDF file and start asking questions about its content.")

    with gr.Row():
        with gr.Column(scale=1):
            file_output = gr.Textbox(label="PDF Processing Status", interactive=False)
            upload_button = gr.UploadButton("Click to Upload a PDF", file_types=[".pdf"])

        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(
                [],
                elem_id="chatbot",
                label="Chat",
                type="messages"
            )
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Please upload a PDF to enable chat",
                container=False,
                interactive=False
            )

    txt.submit(add_text, [chatbot_ui, txt], [chatbot_ui, txt], queue=False).then(
        bot, chatbot_ui, [chatbot_ui, txt]
    )
    upload_button.upload(process_pdf, upload_button, [file_output, txt])

if __name__ == "__main__":
    demo.launch()