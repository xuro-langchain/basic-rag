from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from datastore import retriever

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)



# ========================================================================================================================
# CORE RAG FUNCTIONS
# ========================================================================================================================

@traceable(run_type="retriever")
def retrieve_documents(question: str) -> list:
    """Retrieve documents from vector datastore"""
    print("Retrieving documents...\n")
    # Retrieval
    documents = retriever.invoke(question)
    return documents

@traceable
def generate_response(question: str, documents: list):
    """Generate response using retrieved documents"""
    print("Reviewing documents...\n")
    
    rag_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question} 

    Context: {context} 

    Answer:"""

    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    rag_prompt_formatted = rag_prompt.format(context=formatted_docs, question=question)

    generation = llm.invoke([SystemMessage(content=rag_prompt_formatted), HumanMessage(content=question)])
    return generation



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# ========================================================================================================================
# GUARDRAILS AND REFLECTIONS
# ========================================================================================================================
@traceable
def grade_documents(question: str, documents: list):
    """
    Determines whether the retrieved documents are relevant to the question. Filters documents down to relevant docs
    """
    grade_documents_llm = llm.with_structured_output(GradeDocuments)
    grade_documents_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_documents_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"

    # Score each doc
    filtered_docs = []
    for d in documents:
        grade_documents_prompt_formatted = grade_documents_prompt.format(document=d.page_content, question=question)
        score = grade_documents_llm.invoke(
            [SystemMessage(content=grade_documents_system_prompt)] + [HumanMessage(content=grade_documents_prompt_formatted)]
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
    return filtered_docs


@traceable
def decide_to_generate(filtered_documents: list):
    """
    Determines whether to generate an answer, or to terminate execution if output does not pass guardrails
    """
    if not filtered_documents:
        return False # All documents have been filtered, so we will re-generate a new query
    else:
        return True
    

# ========================================================================================================================
# COMPILED RAG APPLICATION
# ========================================================================================================================

@traceable
def rag(question: str):
    documents = retrieve_documents(question)
    filtered_docs = grade_documents(question, documents)
    
    approved = decide_to_generate(filtered_docs)
    answer = "No relevant documents found. Try a different query."
    if approved:
        answer = generate_response(question, filtered_docs)
    return {"answer": answer}

question = "What is a supervisor architecture?"
answer = rag(question)
print(answer["answer"].content)
