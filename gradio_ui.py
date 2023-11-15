import os
import gradio as gr
from docu_query import HandleRetrival


introduction_str = """# Hello There!\n 
               ### Let me walk you through:"""


with gr.Blocks(title="Document Retriver") as demo:
     with gr.Tab("Upload Documents"):
          with gr.Row():
               gr.Markdown(introduction_str)

          with gr.Row():
               with gr.Column():
                    gr.Markdown("Use the Uplaod button below to select your desired PDF files")
                    upload_button = gr.UploadButton(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"])

                    def upload_files(selected_files):
                         file_path_list = [file.name for file in selected_files]
                         file_output_str = return_output_str(file_path_list)
                         return file_output_str 
                    

                    def return_output_str(paths):
                         output_str = ""
                         for path in paths:
                              output_str += f'PDF Name: {os.path.basename(path)} ({path})\n'
                         return output_str
                    
                    def process_docs(selected_files):
                         if not handle_db_instance.load_documents(selected_files):
                              gr.Error("Could not load documents")
                         else:
                               gr.Error("Successful!")
                    

               with gr.Column():
                    selected_files_output = gr.Textbox(label="Selected PDF files")
                    process_button = gr.Button(value="Process PDFs")
                    upload_button.upload(upload_files, upload_button, selected_files_output)
                    handle_db_instance = HandleRetrival()
                    process_button.click(fn=handle_db_instance.load_documents, inputs=[selected_files_output])
     

     with gr.Tab("Ask question"):
          with gr.Row():
               gr.Markdown("Ask a question related to uplaoded documents")
               
          with gr.Row():
               with gr.Column():
                    question = gr.Textbox(label="Ask question")
                    submit_question = gr.Button(value="Submit")
                    answer_output = gr.TextArea(label="Answer")

                    submit_question.click(fn=handle_db_instance.retriever, inputs=[question], outputs=[answer_output])
          

demo.launch()