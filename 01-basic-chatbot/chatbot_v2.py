import gradio as gr
import ollama

def chat(message, history):
    """
    Chat function with conversation history.

    Args:
        message: Current user message
        history: List of [user_msg, assistant_msg] pairs

    Returns:
        The AI's response
    """
    # Convert Gradio history format to Ollama format
    messages = []

    # Add previous conversation
    for user_msg, assistant_msg in history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': assistant_msg})

    # Add current message
    messages.append({'role': 'user', 'content': message})

    # Get response from Ollama
    response = ollama.chat(
        model='llama3.2',
        messages=messages
    )

    return response['message']['content']

# Create interface
demo = gr.ChatInterface(
    fn=chat,
    title="AI Chatbot with Memory",
    description="I remember our conversation! Try asking follow-up questions.",
    examples=[
        "My name is Alice",
        "What's my name?",
        "Tell me a story",
        "What happened in the story?"
    ]
)

if __name__ == "__main__":
    demo.launch()
