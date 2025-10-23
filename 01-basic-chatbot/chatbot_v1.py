import gradio as gr
import ollama

def chat(message, history):
    """
    Simple chat function that sends message to Ollama and returns response.

    Args:
        message: The user's input message
        history: Previous conversation (not used in v1)

    Returns:
        The AI's response as a string
    """
    # Call Ollama API with the message
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': message}]
    )

    # Extract the response text
    return response['message']['content']

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="My First AI Chatbot",
    description="Ask me anything! Powered by Llama 3.2",
    examples=["What is the capital of France?",
              "Write a haiku about programming",
              "Explain quantum computing in simple terms"]
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
