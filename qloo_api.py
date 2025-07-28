import os
import requests
from dotenv import load_dotenv

# Load your Qloo API key
load_dotenv()
QLOO_API_KEY = os.getenv("QLOO_API_KEY")
print(f"[DEBUG] QLOO_API_KEY: {QLOO_API_KEY}")


QLOO_BASE_URL = "https://api.qloo.com/v1/recommendations"

headers = {
    "Authorization": f"Bearer {QLOO_API_KEY}",   # MUST start with Bearer
    "Content-Type": "application/json"
}


def get_qloo_recommendations(input_terms, domain="music", limit=3):
    """
    :param input_terms: List[str] – User interests or cultural tags
    :param domain: One of ["music", "film", "food", "books", "fashion", "travel"]
    :param limit: Number of recommendations
    :return: List[str] of recommendations
    """
    url = f"{QLOO_BASE_URL}/{domain}"
    payload = {
        "input": input_terms,
        "type": domain,
        "limit": limit
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("recommendations", [])
    except Exception as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"[Qloo Error] Status: {e.response.status_code}, Body: {e.response.text}")
        print(f"[Qloo Error] {e}")
        return []
