from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=1)

data = TextLoader("document loader/notes.txt").load()
tdata = splitter.split_documents(data)
print(tdata,"\n")
print("Number of documents loaded:", len(tdata), "\n")
print("Content of the first document:", tdata[0].page_content, "\n")
