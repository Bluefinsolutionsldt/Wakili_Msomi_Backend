import requests
import json
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8001"
headers = {
    "Content-Type": "application/json"
}

def login(user_id: str, email: str) -> str:
    """Login and get JWT token"""
    response = requests.post(
        f"{BASE_URL}/token",
        json={"username": user_id, "password": "any_password"}
    )
    token = response.json()["access_token"]
    headers["Authorization"] = f"Bearer {token}"
    return token

def create_conversation() -> str:
    """Create a new conversation"""
    response = requests.post(
        f"{BASE_URL}/conversations",
        headers=headers,
        json={
            "language": "en",
            "metadata": {}
        }
    )
    return response.json()["id"]

def ask_question(query: str, conversation_id: str = None) -> dict:
    """Ask a legal question"""
    response = requests.post(
        f"{BASE_URL}/query",
        headers=headers,
        json={
            "query": query,
            "conversation_id": conversation_id,
            "language": "en"
        }
    )
    return response.json()

def list_conversations() -> list:
    """List all conversations"""
    response = requests.get(
        f"{BASE_URL}/conversations",
        headers=headers
    )
    return response.json()

def main():
    # 1. Login
    token = login("edward00", "eddyelly24@gmail.com")
    print(f"Logged in successfully, token: {token[:20]}...")
    
    # 2. Create a conversation
    conversation_id = create_conversation()
    print(f"\nCreated conversation: {conversation_id}")
    
    # 3. Ask a question
    query = "What are the legal requirements for starting a business in Tanzania?"
    response = ask_question(query, conversation_id)
    print(f"\nQuestion: {query}")
    print(f"Answer: {response['response']}")
    print(f"Confidence: {response['confidence_score']}")
    print(f"Processed at: {response['processed_at']}")
    
    # 4. List all conversations
    conversations = list_conversations()
    print("\nYour conversations:")
    for conv in conversations:
        print(f"ID: {conv['id']}")
        print(f"Created at: {conv['created_at']}")
        print(f"Messages: {len(conv['messages'])}")
        print("---")

if __name__ == "__main__":
    main()
