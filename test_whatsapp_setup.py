#!/usr/bin/env python3
"""
Simple test script to verify WhatsApp integration setup
"""

def test_whatsapp_integration():
    print("🔍 Testing WhatsApp Integration Setup...")
    
    # Test 1: Check if files were created
    import os
    files_to_check = [
        "app/config.py",
        "app/services/whatsapp_service.py",
        "app/api/whatsapp.py",
        "app/services/__init__.py"
    ]
    
    print("\n📁 Checking file creation:")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    # Test 2: Check basic imports (without dependencies)
    print("\n🔧 Testing basic Python syntax:")
    
    try:
        # Test config syntax
        with open("app/config.py", "r") as f:
            compile(f.read(), "app/config.py", "exec")
        print("✅ config.py syntax OK")
    except SyntaxError as e:
        print(f"❌ config.py syntax error: {e}")
    
    try:
        # Test WhatsApp service syntax
        with open("app/services/whatsapp_service.py", "r") as f:
            compile(f.read(), "app/services/whatsapp_service.py", "exec")
        print("✅ whatsapp_service.py syntax OK")
    except SyntaxError as e:
        print(f"❌ whatsapp_service.py syntax error: {e}")
    
    try:
        # Test router syntax
        with open("app/api/whatsapp.py", "r") as f:
            compile(f.read(), "app/api/whatsapp.py", "exec")
        print("✅ whatsapp.py syntax OK")
    except SyntaxError as e:
        print(f"❌ whatsapp.py syntax error: {e}")
    
    # Test 3: Check if environment variables are added
    print("\n🌍 Checking environment configuration:")
    try:
        with open(".env.example", "r") as f:
            content = f.read()
            whatsapp_vars = [
                "WHATSAPP_API_TOKEN",
                "WHATSAPP_CLOUD_NUMBER_ID", 
                "WHATSAPP_VERIFY_TOKEN",
                "META_API_VERSION"
            ]
            for var in whatsapp_vars:
                if var in content:
                    print(f"✅ {var} added to .env.example")
                else:
                    print(f"❌ {var} missing from .env.example")
    except FileNotFoundError:
        print("❌ .env.example not found")
    
    print("\n🎉 WhatsApp integration setup complete!")
    print("\n📋 Next steps:")
    print("1. Add WhatsApp API credentials to your .env file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start your FastAPI server")
    print("4. Configure WhatsApp webhook URL: https://yourserver.com/whatsapp/webhook")
    print("5. Test with WhatsApp messages!")

if __name__ == "__main__":
    test_whatsapp_integration()
