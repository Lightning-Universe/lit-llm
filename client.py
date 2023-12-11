import os
import requests
import sys
import textwrap

URL = 'http://127.0.0.1:8000/chat'

LIT_SERVER_API_KEY = os.environ.get("LIT_SERVER_API_KEY", "1234567890")

_HEADERS = {"x-api-key": LIT_SERVER_API_KEY}

def send_prompt(prompt: str):
    res = requests.post(URL, headers=_HEADERS, json={"prompt": prompt, "temperature": 0.1})
    out = res.content.decode()
    print(out)


if __name__ == "__main__":
    prompt = sys.argv[1]
    send_prompt(prompt)