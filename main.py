import os

from dotenv import load_dotenv
from itertools import chain

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.agent import AgentRunResult

import nest_asyncio
nest_asyncio.apply()
### Load environment variables and initialize the language model


load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LOGFIRE_IGNORE_NO_CONFIG'] = '1'

agent = Agent(
    model='openai:gpt-4o-mini',
    system_prompt='You are a helpful AI assistant.',
)

store: dict[str, list[bytes]] = {}

def create_session_if_not_exists(session_id: str) -> None:
    """Makes sure that `session_id` exists in the chat storage."""
    if session_id not in store:
        store[session_id]: list[ModelMessage] = []
    
def get_chat_history(session_id: str) -> list[ModelMessage]:
    """Returns the existing chat history."""
    
    create_session_if_not_exists(session_id)

    # Convert from `bytes` to a list of `Message`s and return the history.
    return list(chain.from_iterable(
        ModelMessagesTypeAdapter.validate_json(msg_group)
        for msg_group in store[session_id]
    ))

def store_messages_in_history(session_id: str, run_result: AgentRunResult[ModelMessage]) -> None:
    """Stores all new messages from the recent `run` with the model, into the local store.

    Receives a session ID and the results that the model returned, fetches all the new 
    messages in `bytes` format and stores them in our local storage.
    """
    create_session_if_not_exists(session_id)

    store[session_id].append(run_result.new_messages_json())
    
def ask_with_history(user_message: str, user_session_id: str) -> AgentRunResult[ModelMessage]:
    """Asks the chatbot the user's question and stores the new messages in the chat history."""

    # Get existing history to send to model
    chat_history = get_chat_history(user_session_id)

    # Ask user's question and send chat history.
    chat_response: AgentRunResult[ModelMessage] = agent.run_sync(user_message, message_history=chat_history)

    # Store new messages in chat history.
    store_messages_in_history(user_session_id, chat_response)

    return chat_response

session_id = 'user_123'

result1 = ask_with_history('Hello! Give me a few ways some people get fooled by AI.', session_id)
print('AI:', result1.data)

result2 = ask_with_history('What was my previous message?', session_id)
print('AI:', result2.data)


print('\nConversation History:')
tmp = get_chat_history(session_id)
for message in get_chat_history(session_id):
    print(f'{message.parts[-1].part_kind}: {message.parts[-1].content}')