import requests
from bs4 import BeautifulSoup
import re
from typing import List

def extract_text_from_url(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download document.")
    
    content_type = response.headers.get("Content-Type", "")
    
    if "text/html" in content_type:
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()
    else:
        text = response.text  # Basic fallback for raw text or plaintext files
    
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    return cleaned_text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
