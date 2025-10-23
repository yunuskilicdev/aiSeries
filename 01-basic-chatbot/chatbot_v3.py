import gradio as gr
import ollama

def chat(message, history, temperature, max_tokens):
    """
    Chat function with configurable parameters.

    Args:
        message: User's message
        history: Conversation history
        temperature: Controls randomness (0.0 - 1.0)
        max_tokens: Maximum length of response

    Returns:
        AI's response
    """
    # Build message history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': assistant_msg})
    messages.append({'role': 'user', 'content': message})

    # Call Ollama with options
    response = ollama.chat(
        model='llama3.2',
        messages=messages,
        options={
            'temperature': temperature,
            'num_predict': max_tokens,
        }
    )

    return response['message']['content']

# Create a simple ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 AI Chatbot with Parameters",
    description="Control AI behavior with temperature and token limits",
    additional_inputs=[
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature"
        ),
        gr.Slider(
            minimum=50,
            maximum=500,
            value=200,
            step=50,
            label="Max Tokens"
        )
    ]
)

if __name__ == "__main__":
    demo.launch()
