import os
from dotenv import load_dotenv
from src.db.database import SessionLocal
from src.generation.generator import generate_answer

load_dotenv()

def run_interactive_qa():
    db = SessionLocal()
    print("==================================================")
    print("     ANNUAL REPORT QA BOT - LIVE TESTER     ")
    print("==================================================")
    print("Ask any question about the company's financials.")
    print("Type 'exit' to quit.")
    
    try:
        while True:
            query = input("\n👤 YOU: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            print("\n🤖 AI is thinking... (Retrieving chunks & generating answer)\n")
            
            try:
                answer = generate_answer(query, db)
                print(f"🤖 BOT:\n{answer}")
                print("\n" + "-" * 50)
            except Exception as e:
                print(f"Error during generation: {e}\n")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        db.close()

if __name__ == "__main__":
    run_interactive_qa()
