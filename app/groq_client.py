import os
import requests

def ask_groq_llm(question):
    try:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            print("Groq API key not found")
            return None

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are EVA2Z assistant. Answer only about GPS trackers, installation, vehicle safety, and related queries. If unrelated, say you don't know."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": 0.3
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print("Groq API error:", response.text)
            return None

    except Exception as e:
        print("Groq Exception:", e)
        return None