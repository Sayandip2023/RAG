import streamlit as st
import torch
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

class DocumentSearchApp:
    def __init__(self):
        self.pdf_file = None
        self.query = None
        self.rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
        self.rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
        self.rag_generator = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", return_dict_in_generate=True)

    def search(self):
        # Convert PDF file to text
        pdf_text = self.pdf_file.getvalue().decode("utf-8")
        
        # Prepare input for the RAG model
        inputs_dict = self.rag_tokenizer.prepare_query(
            query=self.query,
            context=pdf_text
        )

        # Perform retrieval
        with torch.no_grad():
            retriever_results = self.rag_retriever(
                input_ids=inputs_dict["input_ids"].unsqueeze(0),
                attention_mask=inputs_dict["attention_mask"].unsqueeze(0)
            )

        # Generate answer
        with torch.no_grad():
            generated = self.rag_generator.generate(
                input_ids=inputs_dict["input_ids"].unsqueeze(0),
                attention_mask=inputs_dict["attention_mask"].unsqueeze(0),
                retriever_results=retriever_results,
                num_return_sequences=1,
                max_length=100
            )

        # Decode and display the generated answer
        st.write("Search Results:")
        st.write(self.rag_tokenizer.decode(generated[0], skip_special_tokens=True))

def main():
    app = DocumentSearchApp()
    st.title("Interact with your Research Paper using RAG")

    # File upload
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        app.pdf_file = pdf_file

    # Search query input
    app.query = st.text_input("Enter your search query:")

    # Perform search
    if st.button("Search"):
        if app.pdf_file and app.query:
            app.search()
        else:
            st.warning("Please upload a PDF file and enter a search query.")

if __name__ == "__main__":
    main()
