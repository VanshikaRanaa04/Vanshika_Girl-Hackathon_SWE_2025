from groq import Groq

assistant = Groq(api_key="")

conversation = [
    {
        "role": "system",
        "content": (
            "You are a virtual health assistant equipped to offer general wellness tips, symptom insights, lifestyle suggestions, "
            "and awareness based on trustworthy medical knowledge. Your purpose is to help users understand their symptoms better, "
            "but always remind them that this does not replace advice from a licensed healthcare provider."
        )
    },
    {
        "role": "user",
        "content": "I am suffering from fever"
    }
]

response = assistant.chat.completions.create(
    messages=conversation,
    model="llama-3.3-70b-versatile"
)

print(response.choices[0].message.content)
