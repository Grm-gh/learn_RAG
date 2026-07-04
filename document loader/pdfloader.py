from langchain_community.document_loaders import PyPDFLoader



data = PyPDFLoader("document loader/C.pdf").load()
print(data[1].page_content,"\n")
print("Number of documents loaded:", len(data), "\n")