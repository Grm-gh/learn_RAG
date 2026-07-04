from langchain_community.document_loaders import TextLoader
data = TextLoader("document loader/notes.txt").load()
print(data,"\n")
print("Number of documents loaded:", len(data), "\n")
print("Content of the first document:", data[0].page_content, "\n")
