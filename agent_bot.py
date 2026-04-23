from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import json
import os
load_dotenv()
key = os.getenv("AICREDITS_API_KEY")

llm = ChatOpenAI(
    model= "openai/gpt-4o-mini",
    base_url="https://api.aicredits.in/v1",
    api_key=key,
    temperature=0
)


#-------------------RAG SETUP-------------------


loader = TextLoader("autostream_data.md")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    base_url="https://api.aicredits.in/v1",
    api_key=key
)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_query(query: str):
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer based only on the context:

    {context}

    Question: {query}
    """
    return llm.invoke(prompt).content



#-------------------------- Intent classifiar-----------------------------------

def classify_intent(query: str) -> str:
    query = query.lower()

    if any(word in query for word in ["buy", "subscribe", "sign up", "purchase", "get started"]):
        return "lead"

    response = llm.invoke([
        {"role": "system", "content": "You are an intent classifier."},
        {"role": "user", "content": f"""
Classify into: greeting, product, lead.

Query: {query}

Return ONLY one word.
"""}
    ])

    intent = response.content.strip().lower()

    if "greeting" in intent:
        return "greeting"
    elif "lead" in intent:
        return "lead"
    else:
        return "product"


def extract_info(query: str, field: str):
    prompt = f"""
    Extract {field} from the user input.
    If not present, return NONE.

    Input: {query}
    """
    res = llm.invoke(prompt).content.strip()
    return None if res.lower() == "none" else res


def mock_lead_capture(name, email, platform):
    lead = {
        "name": name,
        "email": email,
        "platform": platform
    }

    try:
        with open("leads.json", "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append(lead)

    with open("leads.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"\n[API] Lead captured successfully: {name}, {email}, {platform}\n")



class AgentState(TypedDict):
    messages: List[BaseMessage]
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]



#------------------------------ NODES-------------------------


def router(state: AgentState) -> str:
    query = state["messages"][-1].content

    if state.get("name") is not None or state.get("email") is not None:
        print("DEBUG: Continuing lead flow")
        return "lead"


    state["intent"] = classify_intent(query)

    print("DEBUG INTENT:", state["intent"])
    return state["intent"]


def greeting_node(state: AgentState) -> AgentState:
    print("\nAI: Hey! 👋 How can I help you today?\n")
    return state


def product_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    answer = rag_query(query)
    print(f"\nAI: {answer}\n")
    return state

#--------------------------------Tool Execution (Lead Capture)-------------------------


def lead_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content

    # STEP 1: NAME
    if state.get("name") is None:
        print("\nAI: What's your name?\n")
        state["name"] = ""  
        return state

    if state.get("name") == "":
        state["name"] = query.strip()
        print("\nAI: Could you share your email?\n")
        return state

    if state.get("email") is None:
        import re
        if re.match(r"[^@]+@[^@]+\.[^@]+", query):
            state["email"] = query.strip()
            print("\nAI: Which platform do you create content on?\n")
        else:
            print("\nAI: Please provide a valid email address.\n")
        return state

    if state.get("platform") is None:
        state["platform"] = query.strip()

        mock_lead_capture(
            state["name"],
            state["email"],
            state["platform"]
        )

        print("\nAI: Thanks! You're all set 🚀\n")
        state["intent"] = None
        state["name"] = None
        state["email"] = None
        state["platform"] = None

        return state

    return state


#------------Agent Logic (LangGraph)-------------------


graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("product", product_node)
graph.add_node("lead", lead_node)

graph.add_conditional_edges(
    START,
    router,
    {
        "greeting": "greeting",
        "product": "product",
        "lead": "lead"
    }
)

graph.add_edge("greeting", END)
graph.add_edge("product", END)
graph.add_edge("lead", END)

agent = graph.compile()


#------------------------------memory and execution-------------------------
state = {
    "messages": [],
    "intent": None,
    "name": None,
    "email": None,
    "platform": None
}

user_input = input("User: ")

while user_input != "exit":
    state["messages"].append(HumanMessage(content=user_input))
    state = agent.invoke(state)
    user_input = input("User: ")