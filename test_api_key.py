import os
print("API KEY:", "Found" if os.environ.get("GEMINI_API_KEY") else "Missing")
