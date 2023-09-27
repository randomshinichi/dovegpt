import argparse
from pprint import pprint as pp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('repo_path', type=str, help='path to the repository')
parser.add_argument('--model_path', type=str, default='llama-2-7b-32k-instruct.Q5_K_M.gguf', help='path to the model')
parser.add_argument('--llama', action='store_true', help='use LlamaCpp instead of ChatGPT')

args = parser.parse_args()

repo_path = args.repo_path
model_path = args.model_path

# Load
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*",
    suffixes=[".js"],
    parser=LanguageParser(language=Language.JS, parser_threshold=500) # Currently, the supported languages for code parsing are Python and JavaScript. 
    # Source https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.parsers.language.language_parser.LanguageParser.html
)
documents = loader.load()
# print("# of documents", len(documents))

go_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JS, 
                                                               chunk_size=2000, 
                                                               chunk_overlap=200)
texts = go_splitter.split_documents(documents)
# print("# of split documents (texts)", len(texts))

def chatgpt(texts):
    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 8},
    )

    llm = ChatOpenAI(model_name="gpt-4")
    return db, retriever, llm

def llama(texts) -> LlamaCpp:
    db = Chroma.from_documents(texts, LlamaCppEmbeddings(model_path=model_path))
    retriever = db.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 8},
    )

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=0,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,  
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False)
    return db, retriever, llm

if args.llama:
    db, retriever, llm = llama(texts)
else:
    db, retriever, llm = chatgpt(texts)

memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

print("going to ask some questions now")
questions = [
    "What is the class hierarchy?",
]


while True:
    question = input("Ask a question: ")
    if not question:
        break
    documents = retriever.get_relevant_documents(question)
    pp(documents)
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")