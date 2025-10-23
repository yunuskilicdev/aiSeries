import gradio as gr
import ollama

def chat(message, history, persona_choice, temperature):
    """
    Chat with custom system prompt and personas.

    Args:
        message: User's message
        history: Conversation history
        persona_choice: Selected persona
        temperature: Controls randomness (0.0 - 1.0)

    Returns:
        AI's response
    """
    # Predefined personas
    PERSONAS = {
        "Default": "",
        "Helpful Assistant": "You are a helpful, friendly assistant who provides clear and concise answers.",
        "Pirate": "You are a pirate captain. Respond to everything in pirate speak with 'Arrr!' and nautical terms.",
        "Poet": "You are a poet who responds to everything in verse and rhyme.",
        "Teacher": "You are a patient teacher explaining concepts to a 10-year-old child. Use simple language and examples.",
        "Developer": "You are an expert software developer. Provide technical, precise answers with code examples when relevant."
    }

    messages = []

    # Add system prompt based on persona
    system_prompt = PERSONAS.get(persona_choice, "")
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    # Add conversation history
    for user_msg, assistant_msg in history:
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': assistant_msg})

    # Add current message
    messages.append({'role': 'user', 'content': message})

    response = ollama.chat(
        model='llama3.2',
        messages=messages,
        options={'temperature': temperature}
    )

    return response['message']['content']

# Create a simple ChatInterface with persona support
demo = gr.ChatInterface(
    fn=chat,
    title="🎭 AI Chatbot with Personas",
    description="Choose a persona to change the AI's behavior and personality",
    additional_inputs=[
        gr.Dropdown(
            choices=["Default", "Helpful Assistant", "Pirate", "Poet", "Teacher", "Developer"],
            value="Default",
            label="Choose a Persona"
        ),
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
    demo.launch()
