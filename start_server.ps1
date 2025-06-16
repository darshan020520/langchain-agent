# Set environment variables
$env:OPENAI_API_KEY="your-api-key-here"
$env:SERPAPI_API_KEY="your-serpapi-key-here"

# Start the server
uvicorn api.main:app --reload 