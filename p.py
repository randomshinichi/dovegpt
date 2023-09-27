import argparse
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('repo_path', type=str, help='path to the repository')

args = parser.parse_args()

repo_path = args.repo_path

# Load
loader = DirectoryLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".go"],
    parser=LanguageParser(language=None, parser_threshold=500) # Currently, the supported languages for code parsing are Python and JavaScript. 
    # Source https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.parsers.language.language_parser.LanguageParser.html
)
documents = loader.load()
print("# of documents", len(documents))

go_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.GO, 
                                                               chunk_size=2000, 
                                                               chunk_overlap=200)
texts = go_splitter.split_documents(documents)
print("# of split documents (texts)", len(texts))

db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr", # Also test "similarity"
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(model_name="gpt-4") 
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

print("going to ask some questions now")
questions = [
    "What is the class hierarchy?",
]

for question in questions:
    documents = retriever.get_relevant_documents(question)
    print(documents)
    # result = qa(question)
    # print(f"-> **Question**: {question} \n")
    # print(f"**Answer**: {result['answer']} \n")