# Building Your First AI Agent from Scratch

A step-by-step guide to building a conversational AI agent using Python and the OpenAI API.

## What You'll Build

A weather assistant agent that can:
- Look up current weather for any city
- Convert between temperature units (Fahrenheit, Celsius, Kelvin)
- **Decide on its own** which tool to use based on your questions

This project teaches you the **core pattern** behind every AI agent - from simple chatbots to complex multi-agent systems.

## The Agent Pattern

```
┌─────────────────────────────────────────────────┐
│                  THE AGENT LOOP                 │
│                                                 │
│   1. Gather: messages + tools + system prompt   │
│   2. Call LLM                                   │
│   3. Did LLM request tool calls?                │
│      ├─ YES → Execute tools                     │
│      │        Add results to messages           │
│      │        Go to Step 2                      │
│      └─ NO  → Return the text response          │
│               (Agent is done)                   │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (modern Python package manager)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/timroman234/Building_Your_First_AI_Agent.git
cd Building_Your_First_AI_Agent

# Install dependencies
uv sync

# Create .env file with your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Run the Agent

```bash
uv run main.py
```

### Example Conversations

```
You: What's the weather in Tokyo?
Agent: It's currently 22°C in Tokyo with clear skies...

You: Convert that to Fahrenheit
Agent: 22°C is equal to 71.6°F.

You: What's warmer right now, Paris or London?
Agent: [Fetches both, compares] Paris is warmer at 18°C compared to London's 14°C.
```

## Project Structure

```
├── main.py           # Interactive chat interface
├── agent.py          # The agent loop (core pattern)
├── tools.py          # Tool implementations (weather, conversion)
├── 01_basic_call.py  # Tutorial: basic API call
├── 02_tools_intro.py # Tutorial: understanding tools
├── pyproject.toml    # Project dependencies
└── BUILD_YOUR_FIRST_AI_AGENT.md  # Full tutorial
```

## How It Works

### 1. Tool Definitions
Tools are described to the model so it knows what's available:

```python
{
    "name": "get_weather",
    "description": "Get current weather for a city...",
    "parameters": {...}
}
```

### 2. The Agent Loop
The core pattern that makes an LLM an "agent":

```python
while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        tools=TOOLS,
    )

    if not response.tool_calls:
        return response.content  # Done!

    # Execute tools, add results, loop again
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        conversation_history.append(result)
```

### 3. Tool Execution
When the model requests a tool, we execute it and feed the result back:

```python
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "convert_temperature": convert_temperature,
}
```

## Learn More

See [BUILD_YOUR_FIRST_AI_AGENT.md](BUILD_YOUR_FIRST_AI_AGENT.md) for the complete tutorial with:
- Line-by-line code explanations
- Why each design decision was made
- Common pitfalls and how to avoid them
- Next steps: memory, RAG, multi-agent systems

## Extending the Agent

Adding new tools is straightforward:

1. Write the function in `tools.py`
2. Add the tool definition to `TOOLS` in `agent.py`
3. Register it in `TOOL_FUNCTIONS`

Ideas for new tools: time zones, air quality, calculations, to-do lists, web search.

## Dependencies

- `openai` - OpenAI Python SDK
- `python-dotenv` - Environment variable management
- `requests` - HTTP client for weather API

## License

MIT - Built as a learning project. Extend it, break it, rebuild it.
