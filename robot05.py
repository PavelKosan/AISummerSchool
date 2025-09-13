from llm_utils import openaillm, llmollama

OLLAMA_HOST = ""
OLLAMA_USER = ""
OLLAMA_PASS = ""

image=""
apikey=""

#res=openaillm(image,apikey)
res=llmollama(image, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS)
print("Result JSON")
print(res)

res=llmollama(image, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS, user_prompt="return me just blue and red square object if there is any.")
print("Result JSON")
print(res)

res=llmollama(image, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS, user_prompt="return me just black circle object if there is any.")
print("Result JSON")
print(res)

