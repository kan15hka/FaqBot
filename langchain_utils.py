import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load environment variables
load_dotenv()

googleapi_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = googleapi_key

# Create Google Gen AI LLM Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.6)

# Initialize Google embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load Data from FAQ csv
    loader = CSVLoader(file_path="faqs.csv", source_column="prompt")
    data = loader.load()
    
    # Create embeddings and FAISS instance for storing vector db
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    
    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Check if the vector database exists
    if not os.path.exists(vectordb_file_path):
        print("Vector database does not exist.")
        return

    # Load the vector database from local folder
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    
    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.7})
    
    # Create a prompt template to decrease hallucinations and structure the result
    prompt_template = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer, please structure the response in a natural and conversational tone, as you would respond to a person.
    Try to use full sentences and avoid simply stating 'Yes' or 'No'. If the answer is not found in the context, kindly state 'I don't know.' Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

def get_response(question):
    # Check if the vector database exists
    if os.path.exists(vectordb_file_path):
        chain = get_qa_chain()
        response = chain.invoke({"query": question})
        
        # Check if the response has a valid result
        if "result" in response and response["result"]:
            return response["result"]
        else:
            return "Answer not found."
    else:
        return "Vector database not found. Please create the knowledge base first."



if __name__ == "__main__":
    # Create vector DB if it doesn't exist
    if not os.path.exists(vectordb_file_path):
        create_vector_db()
        print("Vector database created successfully!")
    
    # Test the QA chain
    question = "Do you have internships and EMI Payment?"
    print(f"\nQuestion: {question}")
    print(f"Answer: {get_response(question)}")