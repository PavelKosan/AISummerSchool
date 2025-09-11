from pydantic import BaseModel
import base64
import json
from openai import OpenAI
from ollama import Client
from httpx import DigestAuth
import re

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
class Task05(BaseModel):
    color: str
    shape: str

system_prompt = (
            "Imagine Yourself as a vision assistant for robotics. "
            "Identify the shape of the objects "
            "and its color (red, green, blue). "
            "Output only valid JSON that matches the following schema: "
            '{"shape":"<shape of the object>","color":"<color of the object>"} '
            "Do NOT include any extra text or explanation."
        )

def openaillm(image,apikey):
    client= OpenAI(api_key=apikey)
    encoded_image = encode_image(image)
    response = client.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content" : [
                {"type": "text", "text": "Find the shape and color in this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]}
        ]
    )
    raw = response.choices[0].message.content
    return Task05(**json.loads(raw))


def llmollama(image, server, username, password, model="gemma3:27b"):
    encoded_image = encode_image(image)
    client = Client(
        host=server,
        auth=DigestAuth(username, password)
    )
    response = client.chat(
        model=model,   # multimodal vision model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Find the shape and color in this image.",
            "images": [encoded_image]}
        ]
    )
    raw = response["message"]["content"].strip()
    print("RAW:", repr(raw))

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        json_text = match.group()
        parsed = json.loads(json_text)
        return Task05(**parsed)
    #parsed = json.loads(raw)
    #return Task05(**parsed)
