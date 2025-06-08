import gradio as gr
from src.pdf_processor import extract_text_from_pdf
from src.text_processor import chunk_text
from src.embeddings import VectorStore
from src.qa_system import QASystem

class PDFQA:
    def __init__(self):
        self.vector_store = VectorStore()
        self.qa_system = QASystem()
        self.chunks = []
    
    def load_document(self, pdf_file):
        text = extract_text_from_pdf(pdf_file)
        self.chunks = chunk_text(text)
        self.vector_store.create_embeddings(self.chunks)
        return "Document loaded successfully!"
    
    def ask_question(self, question):
        if not self.chunks:
            return "Please upload a document first"
        
        _, indices = self.vector_store.search(question)
        relevant_context = " ".join([self.chunks[i] for i in indices[0]])
        answer = self.qa_system.answer_question(relevant_context, question)
        return answer

pdf_qa = PDFQA()

with gr.Blocks(title="PDF QA System") as demo:
    gr.Markdown("# PDF Question Answering System")
    gr.Markdown("Upload a PDF and ask questions about its content")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF")
            load_btn = gr.Button("Load Document")
            load_status = gr.Textbox(label="Status")
        with gr.Column():
            question_input = gr.Textbox(label="Your Question")
            ask_btn = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer")
    
    load_btn.click(
        fn=pdf_qa.load_document,
        inputs=file_input,
        outputs=load_status
    )
    
    ask_btn.click(
        fn=pdf_qa.ask_question,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch()
