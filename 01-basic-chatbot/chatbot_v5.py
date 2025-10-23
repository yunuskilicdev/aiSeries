import gradio as gr
import ollama

def chat_stream(message, history, temperature):
    """
    Chat with streaming responses.

    Args:
        message: User's message
        history: Conversation history
        temperature: Controls randomness (0.0 - 1.0)

    Yields:
        Streaming AI response text
    """
    # Build message history
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': assistant_msg})
    messages.append({'role': 'user', 'content': message})

    # Use streaming to yield response chunks
    response_text = ""
    for chunk in ollama.chat(
        model='llama3.2',
        messages=messages,
        options={'temperature': temperature},
        stream=True
    ):
        response_text += chunk['message']['content']
        yield response_text

# Create a simple ChatInterface with streaming
demo = gr.ChatInterface(
    fn=chat_stream,
    title="⚡ Streaming AI Chatbot (v5)",
    description="Watch responses appear in real-time as the AI thinks!",
    additional_inputs=[
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            label="Temperature"
        )
    ]
)

if __name__ == "__main__":
    demo.queue().launch()
