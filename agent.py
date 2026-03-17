"""
agent.py — The agent loop.

This is the core pattern behind every AI agent. The loop:
1. Send messages + tool definitions to the model
2. Check if the model wants to call a tool
3. If yes → run the tool → add the result to messages → go to step 1
4. If no → the model is done, return its text response

This loop can run multiple times in a single conversation turn.
For example: "What's the weather in Tokyo and convert it to Fahrenheit"
might trigger: get_weather → convert_temperature → final answer.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import get_weather, convert_temperature

load_dotenv()
client = OpenAI()


# --- Tool Definitions ---
# These are the descriptions we send to the model.
# They tell the model what tools exist and how to call them.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a given city. "
                "Use this when the user asks about weather, temperature, "
                "or conditions in a specific location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Tokyo' or 'Paris'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": (
                "Convert a temperature value from one unit to another. "
                "Supports Celsius (C), Fahrenheit (F), and Kelvin (K). "
                "Use this when the user asks to convert a temperature."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The temperature value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The source unit: 'celsius', 'fahrenheit', or 'kelvin'"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The target unit: 'celsius', 'fahrenheit', or 'kelvin'"
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]


# --- Tool Dispatcher ---
# This maps tool names (strings) to actual Python functions.
# When the model says "call get_weather", we look up the function here.

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "convert_temperature": convert_temperature,
}


# --- System Prompt ---
# This shapes the agent's personality and behavior.

SYSTEM_PROMPT = """You are a helpful weather assistant. You can:
- Look up current weather for any city in the world
- Convert temperatures between Celsius, Fahrenheit, and Kelvin

When reporting weather, always mention the city name, temperature, and 
conditions. If the user doesn't specify a temperature unit preference, 
use the unit that's standard for the location's country.

Be concise and friendly."""


def run_agent(user_message: str, conversation_history: list = None) -> str:
    """
    Run the agent loop for a single user message.
    
    Parameters:
        user_message: What the user said.
        conversation_history: Previous messages in the conversation.
            If None, starts a fresh conversation.
    
    Returns:
        The agent's final text response.
    
    This function is the entire agent. Everything else is supporting code.
    """
    
    # Initialize or continue the conversation
    if conversation_history is None:
        conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    # Add the user's new message
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # --- The Agent Loop ---
    while True:
        # Step 1: Send everything to the model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            tools=TOOLS,
        )
        
        # Get the model's response message
        assistant_message = response.choices[0].message
        
        # Add the assistant's response to the conversation history.
        # This is crucial — the model needs to see its own previous
        # messages to maintain coherent multi-step reasoning.
        conversation_history.append(assistant_message)
        
        # Step 2: Check if the model wants to call any tools
        if not assistant_message.tool_calls:
            # No tool calls — the model is done thinking and has
            # produced a final text response. Exit the loop.
            return assistant_message.content
        
        # Step 3: The model wants to call one or more tools.
        # Process each tool call.
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            
            # Parse the arguments from JSON string → Python dict
            function_args = json.loads(tool_call.function.arguments)
            
            # Look up and call the actual function
            if function_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[function_name](**function_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {function_name}"})
            
            # Add the tool result to the conversation history.
            # Note: we must include the tool_call_id so the model
            # knows which call this result belongs to.
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        
        # Loop back to Step 1 — send the updated conversation
        # (now including tool results) back to the model.
        # The model will either make more tool calls or produce
        # a final text response.
    
    # Note: This loop naturally handles multi-step reasoning.
    # If the user says "What's the weather in Tokyo in Fahrenheit?",
    # the model might:
    #   1st pass: call get_weather("Tokyo") → gets 25°C
    #   2nd pass: call convert_temperature(25, "celsius", "fahrenheit") → 77°F
    #   3rd pass: return "It's 77°F (25°C) in Tokyo with clear skies!"