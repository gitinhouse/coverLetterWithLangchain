import faulthandler
faulthandler.enable()
from langchain_community.chat_models import ChatLlamaCpp 
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_agent
from langchain.tools import tool

MODEL_PATH = "../models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"


llm = ChatLlamaCpp(
    model_path=MODEL_PATH,
    chat_format="qwen",
    temperature=0.1,
    max_tokens=1024,
    n_ctx=4096, 
    flash_attn=False, 
    n_batch=128,
    n_threads=2,
    n_gpu_layers=0,
    streaming=True,  
    verbose=True 
)
        

system_msg = (
    "You are an expert Cover Letter Writer. Your goal is to write a short, high-conversion Upwork proposal. "
    "\n\n"
    "PORTFOLIO SELECTION LOGIC (CRITICAL):\n"
    "1. **Tech-Stack Priority**: You MUST prioritize projects from the CONTEXT that match the Job's category and Job's Technology (e.g., if the job is 'WordPress', do NOT use 'Shopify' or 'Gmail Extension' links even if they use AI).\n"
    "2. **Keyword Filtering**: Only select URLs where the 'Tech','Category' or 'Details' fields in the CONTEXT share at least one primary keyword with the JOB description.\n"
    "3. **Strict Discard**: If the CONTEXT contains projects that use a completely different platform than the JOB, ignore them. It is better to list only 1 highly relevant URL than 4 irrelevant ones.\n"
    "\n"
    "STRICT FORMATTING RULES FOR GENERATION:\n"
    "1. **No Headers**: Do NOT include [Your Name], [Date], or [Company Address].\n"
    "2. **Opening**: Start with 'Hello,\n"
    "3. **Description**: Include a section like I can develop (e.g,. 'Yes, I can build/modify/develop the website...).\n"
    "4. **Portfolio Section**: Include a section 'You can check some projects , i have worked on:-' "
    "and list 4-5 relevant projects from the context. Each project MUST include exactly these fields:\n"
    "   - Project URL: [URL from context]\n"
    "5. **Closing**: Mention availability for chat and emphasize regular updates.Do NOT include [Your Name].\n"
    "TONE: Direct, technical, and client-focused. Avoid 'fluff' like 'I am writing to express my interest'."
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="global_csv_data"
)

agent = create_agent(
    model=llm,
    # tools=[search_file_context],
    # checkpointer=checkpointer,
    system_prompt=system_msg,
)



@tool
def search_file_context(query: str,thread_id) -> str:
    """Search for global_csv_data in chroma db."""
    try:
        results = vectorstore.similarity_search(query, k=13)
        
        context_parts = []
        for d in results:
            url = d.metadata.get("Project URL","No URL")
            
            context_parts.append(f"Project URL: {url}\n--")
            
        context = "\n".join(context_parts)
        print(f"[CONTEXT ]: context from tool ::{context}")
        return context if context else "No matching data found in the CSV store."
    except Exception as e:
        print(f"Global search error: {e}")
        return "Error accessing CSV data store."


def get_response(question, thread_id=None):
    
    csv_context = search_file_context.invoke({"query": question, "thread_id": thread_id})
    
    final_prompt = (
        f"CONTEXT:\n{csv_context}\n\n"
        f"JOB:\n{question}\n\n"
        "Write the proposal now. Follow the strict Unicode formatting and spacing rules exactly."
    )
    print(f"\nLength of the final prompt : {len(final_prompt)}\n")

    
    def generator():
        print(">>> [DEBUG] Generating optimized response...")
        try:
            for message, metadata in agent.stream(
                {"messages": [("user", final_prompt)]}, 
                stream_mode="messages"
            ):
                if hasattr(message, 'content') and message.content:
                    yield message.content
        except Exception as e:
            print(f"[ERROR] {e}")
            yield "An error occurred."

    return generator()
            
            
def get_all_projects():
    data = vectorstore.get()
    
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