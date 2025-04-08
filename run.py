import uvicorn
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Set default port or use environment variable
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting Career Guidance Assistant on port {port}")
    print("Open your browser and navigate to http://localhost:{port}")
    
    # Run the FastAPI app
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)