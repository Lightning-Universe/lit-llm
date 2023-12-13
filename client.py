import os
import requests
import sys
import textwrap

URL = 'http://127.0.0.1:8000/v1/chat/completions'

LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY", "1234567890")

_HEADERS = {"x-api-key": LIT_SERVER_API_KEY}

def send_prompt(prompt: str):
    messages = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    res = requests.post(URL, headers=_HEADERS, json=messages)
    print(res.json())
    # out = res.content.decode()
    # print(out)


if __name__ == "__main__":
    prompt = sys.argv[1]
    send_prompt(prompt)