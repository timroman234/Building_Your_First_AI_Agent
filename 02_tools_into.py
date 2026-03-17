"""
Step 3: Understanding tool definitions.
We define tools as structured descriptions. The model reads these
descriptions to decide when and how to use each tool.
"""

# This is a tool definition — it describes a function to the model.
# No actual code runs from this. It's purely informational.
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city. Use this when the user asks about weather, temperature, or conditions in a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'San Miguel de Allende' or 'London'"
                }
            },
            "required": ["city"]
        }
    }
}