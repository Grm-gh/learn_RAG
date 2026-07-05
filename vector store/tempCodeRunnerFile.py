result=vectorstore.similarity_search("What is LangChain?", k=2)# k is the number of similar documents to retrieve
for r in result:
    print(r.page_content, r.metadata, "\n")