from llm_utils import openaillm, llmollama

OLLAMA_HOST = "https://ollama.kky.zcu.cz"
OLLAMA_USER = "niryo"
OLLAMA_PASS = "Aec3aiqu3oodahye"

image="C:\\Users\\pajak\\Desktop\\python_p\\AIandRobotics\\outputSHOW.jpg"
apikey="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"

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
