#!/usr/bin/env python3
"""
Interactive testing script for Sheria Kiganjani API
"""
import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SheriaTester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.token = None
        self.current_user = None
        self.current_conversation = None

    def set_user(self, user_id, email=None):
        """Set the current user and get a JWT token"""
        if email is None:
            email = f"{user_id}@example.com"

        # Get JWT token
        response = requests.post(
            f"{self.base_url}/token",
            data={
                "username": user_id,
                "password": "any_password"  # For testing purposes
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )

        if response.status_code != 200:
            print(f"Error setting user: {response.text}")
            return False

        data = response.json()
        self.token = data["access_token"]
        self.current_user = {"user_id": user_id, "email": email}
        return True

    def _get_headers(self):
        """Get headers with JWT token"""
        headers = {
            "Content-Type": "application/json"
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def create_conversation(self, language="en"):
        """Create a new conversation"""
        if not self.token:
            print("Error: Not authenticated. Please set user first.")
            return False

        response = requests.post(
            f"{self.base_url}/conversations",
            json={"language": language},
            headers=self._get_headers()
        )

        if response.status_code != 200:
            print(f"Error creating conversation: {response.text}")
            return False

        self.current_conversation = response.json()["id"]
        return True

    def send_query(self, query):
        """Send a query to the API"""
        if not self.token:
            print("Error: Not authenticated. Please set user first.")
            return False

        if not self.current_conversation:
            print("Creating new conversation...")
            if not self.create_conversation():
                return False

        response = requests.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "conversation_id": self.current_conversation
            },
            headers=self._get_headers()
        )

        if response.status_code != 200:
            print(f"Error sending query: {response.text}")
            return False

        result = response.json()
        print(f"\nYou: {query}\n")
        print(f"A: {result['response']}\n")
        print(f"Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"Processed at: {result.get('processed_at', 'N/A')}")
        return True

    def list_conversations(self):
        """List all conversations for the current user"""
        if not self.token:
            print("Error: Not authenticated. Please set user first.")
            return False

        response = requests.get(
            f"{self.base_url}/conversations",
            headers=self._get_headers()
        )

        if response.status_code != 200:
            print(f"Error listing conversations: {response.text}")
            return False

        conversations = response.json()
        print("\nYour conversations:")
        for conv in conversations:
            print(f"ID: {conv['id']}")
            print(f"Created: {conv.get('created_at', 'N/A')}")
            print(f"Language: {conv.get('language', 'en')}")
            print("---")
        return True

def main():
    tester = SheriaTester()
    
    # Process command line arguments
    if len(sys.argv) < 2:
        print("Usage: python interactive_test.py <command> [options]")
        print("Commands: login, create, list, ask")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "login":
        user_id = None
        email = None
        
        # Parse arguments
        for arg in sys.argv[2:]:
            if arg.startswith("--user-id="):
                user_id = arg.split("=")[1]
            elif arg.startswith("--email="):
                email = arg.split("=")[1]
        
        if not user_id:
            print("Error: --user-id is required")
            sys.exit(1)
            
        if tester.set_user(user_id, email):
            print(f"Logged in as {user_id}" + (f" ({email})" if email else ""))
        
    elif command == "create":
        if tester.create_conversation():
            print(f"Created conversation: {tester.current_conversation}")
            
    elif command == "list":
        tester.list_conversations()
        
    elif command == "ask":
        query = None
        conv_id = None
        
        # Parse arguments
        for arg in sys.argv[2:]:
            if arg.startswith("--query="):
                query = arg.split("=")[1]
            elif arg.startswith("--conversation-id="):
                conv_id = arg.split("=")[1]
                tester.current_conversation = conv_id
        
        if not query:
            print("Error: --query is required")
            sys.exit(1)
            
        tester.send_query(query)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
