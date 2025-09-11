from llm_utils import openaillm, llmollama

OLLAMA_HOST = "https://ollama.kky.zcu.cz"
OLLAMA_USER = "niryo"
OLLAMA_PASS = "Aec3aiqu3oodahye"

image="C:\\Users\\Admin\\Downloads\\niryo\\AISummerSchool\\outputGREENWORKSPACE.jpg"
apikey="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"

#res=openaillm(image,apikey)
res=llmollama(image, OLLAMA_HOST, OLLAMA_USER, OLLAMA_PASS)
print("Result:", res.shape, res.color)

