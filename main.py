import io
import os
import http.client
import pdfkit

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_stream_writer

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

memory = MemorySaver()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_retries=3,
    streaming=True
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
    
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

@tool("pdf_generator", return_direct=True)
def pdf_generator(input: str) -> str:
    """Generates a PDF from the input string."""
    writer = get_stream_writer()
    writer("Generating PDF")
    file_path = "output.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    try:
        pdfkit.from_string(input, output_path='output.pdf')
        print("PDF generated successfully.")
        return "PDF generated successfully."
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return f"Error generating PDF: {str(e)}"
    

@tool("get_comps", return_direct=True)
def get_comps(input:str)->str:
    """Utilizes MLS data from Zillow to get competitors in the market."""
    writer = get_stream_writer()
    writer("Fetching competitors in the market...\n")
    conn = http.client.HTTPSConnection("zillow-com1.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "",
        'x-rapidapi-host': "zillow-com1.p.rapidapi.com"
    }
    
    address = input.replace(" ","%20").replace(",","")
    request_url = f"/propertyComps?address={address}"

    conn.request("GET", request_url, headers=headers)

    res = conn.getresponse()
    data = res.read()
    
    response = llm.invoke("You are a real estate expert. You MUST respond in HTML format that looks highly professional. Analyze the following data and provide a summary of the competitors in the market.\n\n" + data.decode('utf-8'))
    pdf_response = pdf_generator(response.content)
    print("PDF Generation Response:", pdf_response)

    return response

tools = [get_comps, pdf_generator]

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

system_prompt = """
    You are a real estate expert. You will use the get_comps tool to analyze competitors in the market.
    
    ## TOOLS
    get_comps: This tool uses an HTTP request to a real estate API to get the competitors in the market. You MUST format the input as a valid string to pass into a URL.
    pdf_generator: This tool generates a PDF from the input string.
    
    ## INSTRUCTIONS
    1. Make sure the user gave you a valid address. If the address is invalid or unclear, ask the user to clarify or provide a valid address.
    2. Use the get_comps tool ONCE to get the competitors in the market. The tool uses an HTTP request to a real estate API. You MUST format the input as a valid string to pass into a URL.
    3. Continue the conversation until the user explicitly ends it.
"""


while True:
    user_input = input("You: ")
    
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in agent_executor.stream(
        {
            "messages": 
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input)
                    ], 
    
        }, config, stream_mode=["updates"]
        
    ):
        print("Real Estate Chatbot:", chunk)
        print("----")
