from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os


#load_dotenv()
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)
    

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = (
    """
    You are a math-routing assistant with ABSOLUTELY ZERO mathematical, arithmetic, or computation capability.
    You are completely unable to perform even the simplest math in your head. Even basic operations like adding single-digit numbers (e.g., 1 + 1) MUST be sent to your tools.

    CRITICAL RULES FOR MULTI-STEP / COMPOUND EXPRESSIONS:
    1. NEVER perform mental math. Do not calculate, simplify, or solve any part of a mathematical expression in your head.
    2. For compound expressions (e.g., '(90 + 456) / 3' or '(89 + 34) * 9'), you MUST break them down and execute them strictly ONE operation at a time.
    3. You must execute exactly ONE tool call at a time. You must wait for the tool's result before planning or executing the next step.
    4. Never combine steps or skip to the final calculation.
       - Example: To solve '(89 + 34) * 9', your FIRST step must be to call the `addition` tool for '89' and '34'. Only after you receive the tool's response of '123' can you call the `multiplication` tool for '123' and '9'.
    5. DO NOT write out step-by-step mathematical explanations containing the calculated results (e.g., "90 + 456 = 546") in your text response BEFORE those tools have actually executed. You may only state what tool you intend to call next.
    6. If a calculation involves symbolic mathematical constants like 'pi', 'π', or 'e', you must first resolve the numeric string of that constant using the 'get_constantvalue' tool.
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
chat_history = []

async def chat(agent_executor, input_text):
  msg = await agent_executor.ainvoke({"input": input_text, "chat_history": chat_history})
  response = msg['output']
  chat_history.append(HumanMessage(content=input_text))
  chat_history.append(AIMessage(content=response))
  return response

async def main():

    client = MultiServerMCPClient(
    {
        "mcp_server_http": {
            'transport': 'streamable-http',
            'url': 'http://localhost:8050/mcp',
        }
    }
    )

    tools = await client.get_tools()

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        response = await chat(agent_executor, user_input)
        print(f"Assistant: {response}")
    

if __name__ == "__main__":
    asyncio.run(main())
