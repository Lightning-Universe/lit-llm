import os
import requests

URL = 'http://127.0.0.1:8000/v1/chat/completions'

LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY", "1234567890")

_HEADERS = {"x-api-key": LIT_SERVER_API_KEY}


def send_prompt(prompt: str):
    messages = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(URL, headers=_HEADERS, json=messages)
    print(response.json())


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(send_prompt)
