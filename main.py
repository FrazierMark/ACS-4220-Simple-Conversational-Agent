import os
from dotenv import load_dotenv
from itertools import chain
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.agent import AgentRunResult
import nest_asyncio

nest_asyncio.apply()

# Define different personas with their system prompts
PERSONAS = {
    "pirate": {
        "name": "Captain Blackbeard",
        "prompt": "Arr! I be a salty sea dog with centuries of wisdom! I speak in pirate slang, love sharing tales of adventure, and always relate things back to life at sea. I say 'Arr!' frequently and use nautical terms. I be helpful but always stay in character, savvy?"
    },
    "cowboy": {
        "name": "Sheriff Dusty",
        "prompt": "Howdy partner! I'm a good-natured prospecting cowboy from the Old West. I speak with a Western drawl, use lots of cowboy slang, and relate things to life on the range. I often start sentences with 'Well shoot' or 'Reckon', and end with 'partner' or 'pardner'."
    },
    "wizard": {
        "name": "Merlin the Wise",
        "prompt": "Greetings, seeker of knowledge! I am a wise and slightly eccentric wizard who speaks in mystical terms and often relates answers to magical concepts. I pepper my speech with arcane references and magical expressions like 'By the ancient scrolls!' and 'As the crystal ball foretells...'"
    },
    "alien": {
        "name": "Zorp-X9",
        "prompt": "GREETINGS EARTH-BEING! I am a friendly but somewhat confused alien trying to understand human concepts. I occasionally misunderstand Earth idioms, add 'BEEP BOOP' to sentences, and relate everything to my home planet. I speak in an enthusiastic, slightly robotic way."
    },
    "chef": {
        "name": "Chef Pierre",
        "prompt": "Bonjour! I am a passionate and dramatic French chef who relates everything to cooking and food. I sprinkle French phrases into my speech, get extremely excited about culinary topics, and often use cooking metaphors. I say 'Sacrebleu!' when surprised and 'Magnifique!' when pleased."
    }
}

### Load environment variables and initialize the language model
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

# Initialize store and current persona
store: dict[str, list[bytes]] = {}
current_persona = None
agent = None

def clear_chat_history(session_id: str) -> None:
    """Clears the chat history for a given session."""
    if session_id in store:
        store[session_id] = []

def initialize_agent(persona_key: str, session_id: str) -> None:
    """Initialize the agent with a specific persona and clear chat history."""
    global agent, current_persona
    if persona_key not in PERSONAS:
        raise ValueError(f"Invalid persona: {persona_key}")
    
    current_persona = persona_key
    agent = Agent(
        model='openai:gpt-4o-mini',
        system_prompt=PERSONAS[persona_key]["prompt"],
    )
    # Clear chat history when switching personas
    clear_chat_history(session_id)

def create_session_if_not_exists(session_id: str) -> None:
    """Makes sure that `session_id` exists in the chat storage."""
    if session_id not in store:
        store[session_id] = []

def get_chat_history(session_id: str) -> list[ModelMessage]:
    """Returns the existing chat history."""
    create_session_if_not_exists(session_id)
    return list(chain.from_iterable(
        ModelMessagesTypeAdapter.validate_json(msg_group)
        for msg_group in store[session_id]
    ))

def store_messages_in_history(session_id: str, run_result: AgentRunResult[ModelMessage]) -> None:
    """Stores all new messages from the recent `run` with the model, into the local store."""
    create_session_if_not_exists(session_id)
    store[session_id].append(run_result.new_messages_json())

def ask_with_history(user_message: str, user_session_id: str) -> AgentRunResult[ModelMessage]:
    """Asks the chatbot the user's question and stores the new messages in the chat history."""
    if agent is None:
        raise ValueError("Please select a persona first!")

    chat_history = get_chat_history(user_session_id)
    chat_response: AgentRunResult[ModelMessage] = agent.run_sync(user_message, message_history=chat_history)
    store_messages_in_history(user_session_id, chat_response)
    return chat_response

def display_persona_options():
    """Display available personas and return the selected one."""
    print("\nAvailable Personas:")
    for key, persona in PERSONAS.items():
        print(f"- {key}: {persona['name']}")
    
    while True:
        choice = input("\nSelect a persona (type the name): ").lower()
        if choice in PERSONAS:
            return choice
        print("Invalid choice! Please try again.")

def switch_persona(session_id: str) -> None:
    """Handle persona switching including clearing chat history."""
    persona_choice = display_persona_options()
    initialize_agent(persona_choice, session_id)
    print(f"\nSwitched to {PERSONAS[persona_choice]['name']}!")
    print("(Previous conversation history cleared)")

def main():
    session_id = 'user_123'
    
    print("Welcome to the Multi-Persona Chat Bot!")
    
    # Select initial persona
    persona_choice = display_persona_options()
    initialize_agent(persona_choice, session_id)
    
    while True:
        print(f"\nCurrent Persona: {PERSONAS[current_persona]['name']}")
        user_input = input("Enter your message (or 'switch' to change persona, 'quit' to exit): ")
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'switch':
            switch_persona(session_id)
            continue
        
        result = ask_with_history(user_input, session_id)
        print(f"{PERSONAS[current_persona]['name']}:", result.data)

if __name__ == "__main__":
    main()