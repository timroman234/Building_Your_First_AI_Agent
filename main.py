"""
main.py — Interactive chat interface for the weather agent.

This is the user-facing entry point. It runs a simple loop:
ask for input → run the agent → print the response → repeat.

The conversation history persists across turns, so the agent
remembers what you discussed earlier in the session.
"""

from agent import run_agent


def main():
    """Run the interactive chat loop."""
    
    print("=" * 50)
    print("  Weather Agent")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 50)
    print()
    
    # This list persists across the entire session.
    # Each call to run_agent appends to it, so the agent
    # "remembers" the full conversation.
    conversation_history = None
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Run the agent and get a response
            # On the first turn, conversation_history is None,
            # so run_agent will initialize it with the system prompt.
            # After that, it's a list that grows with each turn.
            if conversation_history is None:
                conversation_history = []  # Will be initialized by run_agent
                # Actually, we need to let run_agent initialize it properly:
                conversation_history = None
            
            response = run_agent(user_input, conversation_history)
            
            # After the first call, conversation_history has been
            # populated by run_agent (lists are mutable in Python,
            # so changes inside the function are visible here).
            # On subsequent calls, we need to pass the existing history.
            # But wait — the first call sets conversation_history to None
            # and run_agent creates a NEW list internally...
            #
            # Let's fix this. See the corrected version below.
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Something went wrong. Try again.\n")
            continue
        
        print(f"\nAgent: {response}\n")


# A note on the conversation_history issue above:
# When run_agent receives None, it creates a new list internally.
# But that new list isn't the same object as our local variable.
# To fix this, we need to change our approach slightly.


def main():
    """Run the interactive chat loop (corrected version)."""
    
    print("=" * 50)
    print("  Weather Agent")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 50)
    print()
    
    # Initialize history with the system prompt ourselves,
    # so we own the list object and can pass it on every call.
    from agent import SYSTEM_PROMPT
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = run_agent(user_input, conversation_history)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
