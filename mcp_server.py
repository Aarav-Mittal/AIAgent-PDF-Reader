from fastmcp import FastMCP
import math
import requests
import os

mcp = FastMCP()

def _safe_float(val: str, param_name: str = "input") -> tuple:
    """Safely parses float string and returns (float_val, error_msg)."""
    if val is None:
        return None, f"Error: Argument '{param_name}' is missing."
    clean_val = str(val).strip()
    try:
        return float(clean_val), None
    except ValueError:
        return None, f"Error: Invalid argument '{clean_val}' for '{param_name}'. Arguments MUST be actual numbers (e.g. '123' or '45.6'), NEVER words or variable placeholders like 'result' or 'ans'. You must wait for the output of previous steps and pass that actual number here."

@mcp.tool
def calculate_expression(expression: str) -> str:
    """
    Evaluates a mathematical expression string (e.g. '((76+45)*6)^2', '(90 + 456) / 3 * 2', 'sqrt(144) + 5^2').
    Use this tool FIRST for any mathematical expression, compound formula, or calculation.
    """
    try:
        expr = str(expression).strip()
        expr = expr.replace('^', '**')
        allowed_names = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sqrt': math.sqrt, 'factorial': math.factorial,
            'radians': math.radians, 'degrees': math.degrees,
            'pi': math.pi, 'e': math.e, 'abs': abs, 'pow': pow,
            'log': math.log, 'log10': math.log10, 'exp': math.exp
        }
        res = eval(expr, {"__builtins__": None}, allowed_names)
        return str(res)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"

@mcp.tool
def addition(input1: str, input2: str) -> str:
    """Adds two numbers input1 and input2."""
    a, err1 = _safe_float(input1, "input1")
    if err1: return err1
    b, err2 = _safe_float(input2, "input2")
    if err2: return err2
    return str(a + b)


@mcp.tool
def multiplication(input1: str, input2: str) -> str:
    """Multiplies two numbers input1 and input2."""
    a, err1 = _safe_float(input1, "input1")
    if err1: return err1
    b, err2 = _safe_float(input2, "input2")
    if err2: return err2
    return str(a * b)

@mcp.tool
def subtraction(input1: str, input2: str) -> str:
    """Subtracts input2 from input1."""
    a, err1 = _safe_float(input1, "input1")
    if err1: return err1
    b, err2 = _safe_float(input2, "input2")
    if err2: return err2
    return str(a - b)

@mcp.tool
def division(input1: str, input2: str) -> str:
    """Divides input1 by input2."""
    a, err1 = _safe_float(input1, "input1")
    if err1: return err1
    b, err2 = _safe_float(input2, "input2")
    if err2: return err2
    return str(a / b) if b != 0 else "Cannot divide by zero"

@mcp.tool
def square_root(input: str) -> str:
    """Computes square root of a number."""
    a, err = _safe_float(input, "input")
    if err: return err
    return str(math.sqrt(a)) if a >= 0 else "Cannot find square root of a negative number"

@mcp.tool
def power(input1: str, input2: str) -> str:
    """Computes input1 raised to power input2."""
    a, err1 = _safe_float(input1, "input1")
    if err1: return err1
    b, err2 = _safe_float(input2, "input2")
    if err2: return err2
    return str(a ** b)

@mcp.tool
def factorial(input: str) -> str:
    """Computes factorial of integer input."""
    a, err = _safe_float(input, "input")
    if err: return err
    if not a.is_integer():
        return "Cannot find factorial of a non-integer number"
    a_int = int(a)
    return str(math.factorial(a_int)) if a_int >= 0 else "Cannot find factorial of negative numbers"

@mcp.tool
def degree_to_radian(a: str) -> str:
    """Converts angle degrees to radians."""
    v, err = _safe_float(a, "a")
    if err: return err
    return str(math.radians(v))

@mcp.tool
def sine(angle_radian: str) -> str:
    """Computes sine of angle in radians."""
    v, err = _safe_float(angle_radian, "angle_radian")
    if err: return err
    return str(math.sin(v))

@mcp.tool
def cosine(angle_radian: str) -> str:
    """Computes cosine of angle in radians."""
    v, err = _safe_float(angle_radian, "angle_radian")
    if err: return err
    return str(math.cos(v))

@mcp.tool
def tangent(angle_radian: str) -> str:
    """Computes tangent of angle in radians."""
    v, err = _safe_float(angle_radian, "angle_radian")
    if err: return err
    return str(math.tan(v))

@mcp.tool
def get_constantvalue(const: str) -> str:
    """Gets numeric value of mathematical constant 'pi' or 'e'."""
    name = str(const).strip().lower()
    if name in ("pi", "π"):
        return str(math.pi)
    elif name == "e":
        return str(math.e)
    else:
        return "Invalid constant name"

@mcp.tool
def weather(city: str) -> str:
    """Retrieves weather forecast for a city."""
    api_key = os.getenv('OWM_API_KEY')
    if not api_key:
        return "Error: OWM_API_KEY environment variable is missing or empty."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        Condition = data['weather'][0]['description']
        return f"Temperature: {temp}°C\nHumidity: {humidity}%\nCondition: {Condition}"
    else:
        return f"Error fetching weather data: {data.get('message', 'Unknown error')}"


CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def _get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@mcp.tool
def index_pdf(pdf_path: str) -> str:
    """Indexes PDF document at pdf_path into Chroma vector database."""
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at path '{pdf_path}'."
    if not pdf_path.lower().endswith('.pdf'):
        return f"Error: File '{os.path.basename(pdf_path)}' is not a PDF file."

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        if not pages:
            return "Error: Could not extract any text from the PDF file."
        
        full_content = "\n\n".join([p.page_content for p in pages])
        separators = [
            r"\n\nChapter\s+\d+(\.\d+)*[\s\S]*?(?=\n\nChapter\s+\d+(\.\d+)*|$)",
            r"\n\s*\d+\.\d+\s+[^\n]*",
            r"\n\s*\d+\s+[^\n]*",
            "\n\n\n", "\n\n", "\n", " ", "",
        ]
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=True
        )
        docs = text_splitter.create_documents([full_content])
        embeddings = _get_embeddings_model()
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PERSIST_DIR)
        return f"Successfully indexed '{os.path.basename(pdf_path)}' with {len(docs)} chunks into ChromaDB."
    except Exception as e:
        return f"Error indexing PDF: {str(e)}"

@mcp.tool
def search_pdf_knowledge(query: str, k: str = "5") -> str:
    """Searches indexed PDF document in ChromaDB for relevant knowledge chunks."""
    try:
        from langchain_community.vectorstores import Chroma
        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            return "Error: No PDF has been indexed yet. Please upload and index a PDF first."
        
        embeddings = _get_embeddings_model()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        num_k = int(k) if k.isdigit() else 5
        docs = vectorstore.similarity_search(query, k=num_k)
        if not docs:
            return f"No relevant content found in PDF for query: '{query}'."
        results = [f"--- Chunk {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(results)
    except Exception as e:
        return f"Error searching PDF: {str(e)}"

@mcp.tool
def summarize_pdf_topic(topic: str) -> str:
    """Retrieves PDF document text chunks for topic summary."""
    try:
        from langchain_community.vectorstores import Chroma
        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            return "Error: No PDF has been indexed yet."
        embeddings = _get_embeddings_model()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        docs = vectorstore.similarity_search(topic, k=6)
        if not docs:
            return f"No relevant content found for topic: '{topic}'."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool
def generate_pdf_notes(topic: str) -> str:
    """Retrieves PDF document text chunks for creating study notes."""
    try:
        from langchain_community.vectorstores import Chroma
        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            return "Error: No PDF has been indexed yet."
        embeddings = _get_embeddings_model()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        docs = vectorstore.similarity_search(topic, k=6)
        if not docs:
            return f"No relevant content found for topic: '{topic}'."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool
def generate_pdf_questions(topic: str) -> str:
    """Retrieves PDF document text chunks for practice questions."""
    try:
        from langchain_community.vectorstores import Chroma
        if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
            return "Error: No PDF has been indexed yet."
        embeddings = _get_embeddings_model()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        docs = vectorstore.similarity_search(topic, k=6)
        if not docs:
            return f"No relevant content found for topic: '{topic}'."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8050)
