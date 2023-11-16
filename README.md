# docu-query

Retrieval-Augmented Generation (RAG) Application.

Tools used:
- LangChain
- Hugging Face
- Deeplake
- Gradio

Application Flow:
- User uploads PDF document(s).
- Vector embeddings using HuggingFaceEmbeddings are created.
- Vector embeddings are stored locally using the vectorDB Deeplake.
- Users submits a question/query.
- Retrival to the vectorDB is issued to get more context. This contextual data is along side used the user prompt (context enrichment).
- The query and contextual data are passed to the LLM.
- The LLM responds back based on the context.
- Resopnse is displayed to user.

