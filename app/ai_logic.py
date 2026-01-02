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
    "You are an expert Upwork Proposal Writer. Your goal is to generate professional, "
    "direct proposals. You must never include step numbers (1, 2, 3) in your final response. "
    "You must follow the structure provided in the user prompt exactly without skipping sections."
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
def search_file_context(tech: str, query: str,thread_id) -> str:
    """Search for global_csv_data in chroma db."""
    try:
        combined_search = f"{tech} {query}"
        results = vectorstore.similarity_search(combined_search, k=9)
        
        context_parts = []
        for d in results:
            url = d.metadata.get("Project URL","No URL")
            categories = d.metadata.get("Categories", "N/A") 
            
            context_parts.append(f"Project URL: {url}\nCategories: {categories}\n--")
            
        context = "\n".join(context_parts)
        print(f"[CONTEXT ]: context from tool for {tech}::{context}")
        return context if context else "No matching data found in the CSV store."
    except Exception as e:
        print(f"Global search error: {e}")
        return "Error accessing CSV data store."


def get_response(question,tech_list, thread_id=None):
    
    if not isinstance(tech_list, list):
        tech_list = [tech_list] if tech_list else []
        
    structured_context = ""
    
    for i in tech_list:
        print(f" >>> Searching for : {i}")
        projects = search_file_context.invoke({"tech":i,"query":question,"thread_id":thread_id})  
        
        structured_context += f"TECHNOLOGY: {i}\n"
        structured_context += f"RELEVANT PROJECTS : \n{projects}\n"
        structured_context += "---\n"
    
    # final_prompt = (
    #     f"CONTEXT (Categorized by Technology):\n{structured_context}\n"
    #     f"JOB:\n{question}\n\n"
    #     "TASK:\n"
    #     "Write the proposal. For EACH technology listed in the CONTEXT , if the context specifies plugins : get plugins ::else get projects,create a section: "
    #     " 'I have worked with [Technology Name] and built these projects/plugins:' "
    #     "followed by 3-4 project or plugin URLs from that specific Category. " 
    #     "If less than 3 projects exist in a category , list only the relevent 1 -2 projects"
    # )
    final_prompt = (
        f"CONTEXT (Categorized by Technology):\n{structured_context}\n"
        f"JOB DESCRIPTION :\n{question}\n\n"
        "TASK: Write a proposal based on the CONTEXT and JOB above.\n"
        "Follow this exact sequence:\n"
        "1. Start with exactly 'Hello,'\n"
        "2. Write a 3-4 sentence technical paragraph explaining how you will solve the JOB requirements. Start with 'Yes, I can...'\n"
        "3. For EACH technology listed in the CONTEXT , if the context specifies plugins : get plugins ::else get projects,write a section :"
        "'I have worked with [Technology Name] and built these projects:' "
        "followed by 3-4 project or plugin URLs and project categories from that specific project. " 
        "If less than 3 projects exist in a category , list only the relevent 1 -2 projects .\n"
        "4. End with a professional closing paragraph regarding your experience."
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