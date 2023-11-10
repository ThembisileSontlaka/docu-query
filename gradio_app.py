import gradio as gr
from rag import load_documents

with gr.Blocks(title="Document Retriver") as demo:
     with gr.Tab("Upload Documents"):
          with gr.Row():
               with gr.Column():
                    gr.Markdown("Use the Uplaod button below to select your desired PDF files")
                    upload_pdfs_button = gr.UploadButton(str="Upload PDF Files", file_count="multiple", file_types=[".pdf"])
                    
                    def upload_file(selecetd_pdfs):
                         file_paths = [pdf.name for pdf in selecetd_pdfs]
                         return file_paths
                    
               with gr.Column():
                    file_output = gr.Textbox(label="Selected PDF files")
                    process_button = gr.Button(value="Process PDFs")

                    upload_pdfs_button.upload(upload_file, upload_pdfs_button, file_output)
                    process_button.click(load_documents, inputs=file_output)
demo.launch()