import os

from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate



data = PyPDFLoader("document loader/C.pdf").load()
template = ChatPromptTemplate.from_messages(
    [("system", "you are a ai that summarizes the text."),
     ("human", "{data}")]
)
prompt = template.format(data=data[1].page_content)


chat = ChatMistralAI(
    model="mistral-small-2603",
)
result=chat.invoke(prompt)
print(result.content)
