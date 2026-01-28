import json
import requests
from bs4 import BeautifulSoup

MAX_BYTES = 1_000_000  # 1 MB limit

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts & styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def lambda_handler(event, context):
    """
    Expected input from Bedrock tool:
    {
      "url": "https://example.com"
    }
    """

    url = event.get("url")
    if not url:
        return {"error": "URL is required"}

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Bedrock-Web-Crawler-Agent"},
            stream=True,
            allow_redirects=True
        )

        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            if len(content) > MAX_BYTES:
                break

        html = content.decode(response.encoding or "utf-8", errors="ignore")
        text = clean_html(html)

        return {
            "url": url,
            "text": text[:5000]  # truncate response
        }

    except Exception as e:
        return {"error": str(e)}
