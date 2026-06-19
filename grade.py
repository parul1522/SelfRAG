
import os.path
from os import listdir
from os.path import isfile,join
from typing import List,TypedDict,Literal,get_args
from pydantic import BaseModel,Field
import time
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY =os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def process_document(file_path):
    if os.path.isfile(file_path):
        if file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith((".png",".jpeg","jpg")):
            loader = UnstructuredImageLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_path}"
            )
        docs = loader.load()
    else:
        docs = [Document(page_content=file_path, metadata={"source": "pasted_text"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

class State(TypedDict):
    question:str
    original_question: str
    docs: List[Document]
    answer:str
    retry_count:int
    vectorstore:object

class GradeDoc(BaseModel):
    score: Literal["yes","no"]=Field(
        description="Is this document relevant?"
    )

def grade_doc(question,document):
    structured_llm = llm.with_structured_output(GradeDoc)
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader.Return yes only if the document contains information needed to answer the question.Otherwise return no."),
    ("human", "Question: {question}\nDocument: {document}")
    ])
    chain = prompt | structured_llm
    return chain.invoke({"question":question,"document":document})

def grade_hallucination(output,document):
    structured_llm = llm.with_structured_output(GradeDoc)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a strict fact-checker.

        Return YES only if EVERY factual claim in the answer
        is directly supported by the document or can be derived
        from it using simple arithmetic.

        Return NO if:
        - any fact is missing from the document
        - any assumption is made
        - any outside knowledge is used
        - any hypothetical example is introduced

        Be extremely strict.
        """),

        ("human",
        """
        Document:
        {document}

        Answer:
        {output}
        """)
    ])
    chain = prompt | structured_llm
    return chain.invoke({"output":output,"document":document})

def grade_answer(question,answer):
    structured_llm = llm.with_structured_output(GradeDoc)
    prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    Return YES if:

    - the answer correctly answers the question

    OR

    - the answer says "I don't know"
      and the information is genuinely missing.

    Return NO otherwise.
    """
    ),
    ("human",
    """
    Question: {question}

    Answer: {answer}
    """
    )
    ])
    chain = prompt | structured_llm
    return chain.invoke({"question":question,"answer":answer})

def retrieve(state:State):
    question = state["question"]
    vectorstore=state.get("vectorstore")
    if vectorstore is None:
        return {"docs": []}
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(question)
    return {"docs":retrieved_docs}

def generate(state:State):
    question = state["question"]
    docs = state["docs"]
    context = "\n\n".join([f"Source {i+1}: {d.page_content}"
               for i,d in enumerate(docs)])
    prompt = f"""
                    You are a RAG assistant.

                    Use ONLY the information in the context.

                    Rules:
                    1. Never use outside knowledge.
                    2. Never make assumptions.
                    3. Never invent facts.
                    4. You may combine facts from multiple chunks.
                    5. You may do arithmetic.
                    6. You may do comparisons.

                    If the answer cannot be determined from the context,
                    respond with exactly:

                    I don't know

                    Context:
                    {context}

                    Question:
                    {question}
                    """
    result = llm.invoke(prompt)
    return {"answer": result.content}

def rewrite_query(state:State):
    question = state["original_question"]
    result = llm.invoke(f"""You are an expert query rewriter.
                        Rewrite the question to improve semantic search retrieval.
                        Keep the original meaning.Return only the rewritten question.
                        Question :{question}""")
    return {"question": result.content,"original_question":state["original_question"],"retry_count": state["retry_count"]+1,"docs":state["docs"]}

def decide_node(state : State):
    if not state["docs"]:
        if state["retry_count"]>=2:
            return "generate"
        return "rewrite"
    return "generate"

def decide_after_hallucination(state:State):
    answer = state["answer"]
    docs = state["docs"]
    if answer.strip().lower() == "i don't know":
            return "grade_answer"
    context = "\n\n".join([d.page_content for d in docs])
    result = grade_hallucination(answer, context)
    if result.score == "no":  
        if state["retry_count"]>=2:
            return "grade_answer"
        return "rewrite"
    return "grade_answer"


def decide_after_answer(state:State):
    question=state["original_question"]
    answer = state["answer"]
    result=grade_answer(question,answer)
    if result.score =="no":
        if state["retry_count"]>=3:
            return "end"
        return "rewrite"
    return "end"

def grade_docs_node(state):
    return {"docs": state["docs"]}

def grade_answer_node(state: State):
    return state

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_docs", grade_docs_node)
graph.add_node("rewrite", rewrite_query)
graph.add_node("generate", generate)

graph.add_edge(START,"retrieve")
graph.add_edge("retrieve","grade_docs")
graph.add_conditional_edges("grade_docs", decide_node, {"rewrite": "rewrite", "generate": "generate"})
graph.add_edge("rewrite","retrieve")
graph.add_conditional_edges("generate", decide_after_hallucination, {"rewrite":"rewrite","grade_answer":"grade_answer"})
graph.add_node("grade_answer", grade_answer_node)
graph.add_conditional_edges("grade_answer",decide_after_answer,{"rewrite":"rewrite","end":END})

grap = graph.compile()
