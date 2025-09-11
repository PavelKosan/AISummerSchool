from pydantic import BaseModel
import base64
import json
from openai import OpenAI
import ollama
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
class Task05(BaseModel):
    color: str
    shape: str


image="C:\\Users\\Admin\\Downloads\\niryo\\AISummerSchool\\outputGREENWORKSPACE.jpg"

system_prompt = (
            "Imagine Yourself as a vision assistant for robotics. "
            "Identify the shape of the objects "
            "and its color (red, green, blue). "
            "Output only valid JSON that matches the following schema: "
            '{"shape":"<shape of the object>","color":"<color of the object>"} '
            "Do NOT include any extra text or explanation."
        )


def openaillm(image):
    client= OpenAI(api_key="")

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
    parsed = Task05(**json.loads(raw))
    print("Parsed result:", parsed.shape, parsed.color)



openaillm(image)


