"""
Generate test JWT tokens for API testing
"""
from jose import jwt
import datetime

# Secret key for token signing (must match the one in auth.py)
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

def generate_test_token(user_id: str, email: str) -> str:
    """Generate a test JWT token"""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

if __name__ == "__main__":
    # Generate tokens for test users
    user1_token = generate_test_token("user1", "user1@example.com")
    user2_token = generate_test_token("user2", "user2@example.com")
    
    print("\nTest Tokens (valid for 24 hours):")
    print("\nUser 1 Token:")
    print(user1_token)
    print("\nUser 2 Token:")
    print(user2_token)
    print("\nAdd these tokens to your test_user_conversations.sh script")
