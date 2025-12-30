import os
import uuid
from psycopg_pool import ConnectionPool
from langchain_community.chat_models import ChatLlamaCpp 
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/langchain"
MODEL_PATH = "../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

pool = ConnectionPool(
    conninfo=DATABASE_URL, 
    max_size=20,
    kwargs={"autocommit": True}
)

llm = None
embeddings = None
vectorstore = None
agent = None


def get_vectorstore():
    global embeddings, vectorstore
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if vectorstore is None:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="global_csv_data"
        )
    return vectorstore


@tool
def search_file_context(query: str,thread_id) -> str:
    """Search for global_csv_data in chroma db."""
    try:
        vs = get_vectorstore()
        results = vs.similarity_search(query, k=6)
        
        context_parts = []
        for d in results:
            url = d.metadata.get("Project URL","No URL")
            tech = d.metadata.get("Technology","N/A")
            desc = d.metadata.get("Description",d.page_content)
            
            context_parts.append(f"Project URL: {url}\nTech: {tech}\nDetails: {desc}\n--")
            
        context = "\n".join(context_parts)
        print(f"[CONTEXT ]: context from tool ::{context}")
        return context if context else "No matching data found in the CSV store."
    except Exception as e:
        print(f"Global search error: {e}")
        return "Error accessing CSV data store."


def initialize_llm():
    global llm , agent 
    if llm is None:
        llm = ChatLlamaCpp(
            model_path=MODEL_PATH,
            chat_format="qwen",
            temperature=0.1,
            max_tokens=1024,
            n_ctx=4096, 
            n_threads=4,  
            n_batch=64,      
            n_gpu_layers=0,
            streaming=True,  
            verbose=False
        )
        
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        system_msg = (
            "You are an expert Cover Letter Writer. Your goal is to write a short, high-conversion Upwork proposal. "
            "\n\n"
            "STRICT FORMATTING RULES FOR GENERATION:\n"
            "1. **No Headers**: Do NOT include [Your Name], [Date], or [Company Address].\n"
            "2. **Opening**: Start with 'Hello,\n"
            "3. **Description**: Include a section like I can develop (e.g,. 'Yes, I can build/modify/develop the website...).\n"
            "4. **Portfolio Section**: Include a section 'You can check some projects , i have worked on:-' and list 3-4 relevant URLs from the context provided with a small desctiption about the project.\n"
            "5. **Queries Section**: Include a section labeled '=> `Kindly clarify some queries`:-' followed by 2-3 specific technical questions based on the Job Description.\n"
            "6. **Skills Section**: Use the '➤' emoji for a bulleted list of technical skills and start every skill in a new line (e.g., ➤ I am skilled in...).\n"
            "7. **Closing**: Mention availability for chat and emphasize regular updates.Do NOT include [Your Name].\n"
            "8. **No Footer**: Do NOT include [Your Name].\n\n"
            "TONE: Direct, technical, and client-focused. Avoid 'fluff' like 'I am writing to express my interest'."
        )


        agent = create_agent(
            model=llm,
            # tools=[search_file_context],
            # checkpointer=checkpointer,
            system_prompt=system_msg,
        )

def get_response(question, thread_id=None):
    initialize_llm()
    
    search_query = question.split("\n\n")[0] if "\n\n" in question else question
    
    csv_context = search_file_context.invoke({"query": question, "thread_id": thread_id})
    
    final_prompt = (
        f"CONTEXT:\n{csv_context}\n\n"
        f"JOB:\n{question}\n\n"
        "Write the proposal now. Follow the strict Unicode formatting and spacing rules exactly."
    )
    print(f"\nLength of the final prompt : {len(final_prompt)}\n")

    config = {"configurable": {"thread_id": thread_id}}
    
    def generator():
        print(">>> [DEBUG] Generating optimized response...")
        try:
            for message, metadata in agent.stream(
                {"messages": [("user", final_prompt)]}, 
                config, 
                stream_mode="messages"
            ):
                if hasattr(message, 'content') and message.content:
                    yield message.content
        except Exception as e:
            if "context window" in str(e).lower():
                print("[CRITICAL] Context limit hit. Wiping session history...")
                clear_session_history(thread_id)
                yield "Error: History cleared due to memory limits. Please try your question again."
            else:
                print(f"[ERROR] {e}")
                yield "An error occurred."

    return generator()
            
            
def clear_session_history(thread_id):
    """Officially wipes a LangGraph thread using the checkpointer."""
    global agent
    if agent is None:
        initialize_llm()
    
    try:
        checkpointer = agent.checkpointer
        if checkpointer:
            checkpointer.delete_thread(thread_id)
            print(f"DEBUG: Successfully deleted thread {thread_id} from Postgres.")
    except Exception as e:
        print(f"DEBUG: Failed to delete thread: {e}")

def get_all_projects():
    vs = get_vectorstore()
    data = vs.get()
    
    ids = data.get('ids', [])
    metadatas = data.get('metadatas', [])
    documents = data.get('documents', [])
    
    print(f"DEBUG: IDs: {len(ids)}, Meta: {len(metadatas)}, Docs: {len(documents)}")
    
    formatted_results = []
    
    
    for id_val, meta , doc in zip(ids,metadatas , documents):
        try:
            m = meta if meta is not None else {}
            
            record = {
                "key": id_val,
                "content": doc,
                "url": m.get("Project URL") or m.get("url") or "#",
                "categories": m.get("Categories") or "N/A",
                "tech": m.get("Technology") or "N/A",
                "priority": m.get("Priority") or "0",
                "description": m.get("Description") or doc[:150],
            }
            formatted_results.append(record)
        except Exception as e:
            print(f"Row skip error: {e}")
            continue
    
    print(f"DEBUG: Successfully formatted {len(formatted_results)} records")
    return formatted_results