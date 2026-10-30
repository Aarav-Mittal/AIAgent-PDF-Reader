@echo off
echo Starting FastMCP Server and Streamlit UI...
start "FastMCP Server" uv run python mcp_server.py
timeout /t 3 /nobreak >nul
uv run streamlit run app.py
