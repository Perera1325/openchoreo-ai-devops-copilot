import ollama

def generate_response(question: str) -> str:
    """
    Sends the question to local Ollama and returns the generated response.
    Model used: gemma3:4b
    """
    try:
        response = ollama.generate(model='gemma3:4b', prompt=question)
        return response['response']
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return f"Error: Could not generate response. Details: {e}"
