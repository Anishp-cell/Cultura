import requests

def query_llama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.2",  # Make sure this model is pulled (see below)
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        if "response" in result:
            return result["response"]
        else:
            print("⚠️ No 'response' key. Here's the full result:")
            print(result)
            return "⚠️ Error: LLaMA did not return a valid response."

    except requests.exceptions.RequestException as e:
        print("❌ HTTP Error:", e)
        return "❌ LLaMA server error"

