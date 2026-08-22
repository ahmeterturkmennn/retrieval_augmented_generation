import ollama
def generate_answer(prompt):

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}  
    )
    return response["message"]["content"]