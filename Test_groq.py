#!/usr/bin/env python3
"""
Simple test script to verify Groq API connection
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if Groq API key exists
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("❌ GROQ_API_KEY not found in .env file")
    print("🔧 Make sure your .env file contains:")
    print("   GROQ_API_KEY=your_key_here")
    exit(1)

print(f"✅ Found Groq API key (length: {len(groq_key)})")

# Test 1: Import phi's Groq class
try:
    from phi.model.groq import Groq
    print("✅ Successfully imported phi.model.groq.Groq")
except ImportError as e:
    print(f"❌ Failed to import Groq: {e}")
    print("🔧 Try: pip install --upgrade phidata")
    exit(1)

# Test 2: Create Groq model instance
try:
    model = Groq(id="llama3-70b-8192", api_key=groq_key)
    print("✅ Created Groq model instance")
except Exception as e:
    print(f"❌ Failed to create Groq model: {e}")
    exit(1)

# Test 3: Make a simple API call with proper message format
try:
    print("🧪 Testing API call...")
    
    # Try different ways to invoke the model
    try:
        # Method 1: Simple string (what we tried before)
        response = model.invoke("Say hello!")
        print("✅ API call successful with simple string!")
        print(f"📝 Response: {response}")
    except Exception as e1:
        print(f"⚠️ Simple string method failed: {e1}")
        
        try:
            # Method 2: With message format
            from phi.agent import Agent
            agent = Agent(model=model)
            response = agent.run("Say hello!")
            print("✅ API call successful with Agent!")
            print(f"📝 Response: {response}")
        except Exception as e2:
            print(f"⚠️ Agent method failed: {e2}")
            
            try:
                # Method 3: Direct model call with proper format
                response = model.generate("Say hello!")
                print("✅ API call successful with generate!")
                print(f"📝 Response: {response}")
            except Exception as e3:
                print(f"❌ All methods failed:")
                print(f"   String invoke: {e1}")
                print(f"   Agent: {e2}")
                print(f"   Generate: {e3}")
                raise e1
                
except Exception as e:
    print(f"❌ API call failed: {e}")
    print("🔧 This might be a phi library version issue")
    
    # Check if it's a rate limit error
    if "rate limit" in str(e).lower():
        print("⚠️ Rate limit error - wait a moment and try again")
    elif "401" in str(e):
        print("⚠️ Authentication error - check your API key")
    elif "quota" in str(e).lower():
        print("⚠️ Quota exceeded - check your Groq account")
    elif "role" in str(e):
        print("⚠️ Message format issue - this is a known phi library issue")
        print("🔧 Try updating phi: pip install --upgrade phidata")
    
    # Don't exit, let's try to fix the main script anyway
    print("\n⚠️ Groq connection has issues, but let's try the main script with Agent wrapper")
    
print("\n🔧 Updating main script to use Agent wrapper instead of direct model calls...")