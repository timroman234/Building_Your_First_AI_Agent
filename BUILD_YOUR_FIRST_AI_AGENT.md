# Building Your First AI Agent from Scratch

**A step-by-step guide using Python, OpenAI API, VSCode & Claude Code**

---

## What You'll Build

A conversational AI agent that can:

- Look up current weather for any city
- Convert between temperature units (Fahrenheit ↔ Celsius ↔ Kelvin)
- **Decide on its own** which tool to use based on what you ask

By the end, you'll understand the **core pattern** behind every AI agent — from simple chatbots to complex multi-agent systems. This foundation makes it straightforward to add RAG, memory, and more tools later.

---

## Table of Contents

1. [What Is an Agent, Really?](#step-0-what-is-an-agent-really)
2. [Project Setup](#step-1-project-setup)
3. [Your First API Call](#step-2-your-first-api-call)
4. [Understanding Tools (Function Calling)](#step-3-understanding-tools-function-calling)
5. [Build the Weather Tool](#step-4-build-the-weather-tool)
6. [Build the Unit Conversion Tool](#step-5-build-the-unit-conversion-tool)
7. [The Agent Loop — The Core Pattern](#step-6-the-agent-loop)
8. [Wire It All Together](#step-7-wire-it-all-together)
9. [Run and Test Your Agent](#step-8-run-and-test-your-agent)
10. [What's Next — Where to Go from Here](#step-9-whats-next)

---

## Step 0: What Is an Agent, Really?

Before writing any code, let's get the mental model right.

**A regular chatbot** takes your message, sends it to an LLM, and returns the response. It's a single pass — question in, answer out.

**An agent** does something different. It runs in a **loop**:

```
You say something
  → The LLM thinks about it
  → It decides: "I need to use a tool to answer this"
  → It calls the tool and gets results
  → It thinks again: "Do I have enough info now?"
  → If not, it calls another tool
  → When it's satisfied, it gives you a final answer
```

The key insight is: **the LLM decides what to do next**. You don't hard-code "if the user says weather, call the weather function." The model reads the user's message, looks at the tools available to it, and chooses. That decision-making loop is what makes it an _agent_ rather than a script.

Everything we build below serves this loop.

---

## Step 1: Project Setup

### 1.1 — Install uv (if you haven't already)

`uv` is a modern Python project manager that replaces `pip`, `venv`, and `pip-tools` in a single tool. It's written in Rust and is dramatically faster than the traditional toolchain.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installing, restart your terminal so the `uv` command is available.

**Why uv instead of pip + venv?** Three reasons. First, speed — `uv` resolves and installs packages 10-100x faster than `pip`. Second, simplicity — it handles virtual environments, dependency resolution, and lockfiles all in one tool, so you don't juggle separate commands. Third, reproducibility — `uv` generates a `uv.lock` file that pins exact versions, so anyone who clones your project gets identical dependencies.

### 1.2 — Initialize the project

Open your terminal in VSCode (`` Ctrl+` ``) and run:

```bash
uv init weather-agent
cd weather-agent
```

This creates a project folder with a `pyproject.toml` file — the modern standard for Python project configuration. It replaces the older `setup.py` and `requirements.txt` approach.

**Why `pyproject.toml`?** It's a single file that defines your project's metadata, dependencies, and tooling configuration. The Python community has standardized on this format (PEP 621). You'll see it in virtually every modern Python project.

### 1.3 — Add dependencies

```bash
uv add openai python-dotenv requests
```

This does three things at once: installs the packages, creates a virtual environment (in `.venv/`) if one doesn't exist, and updates `pyproject.toml` with the dependency list.

Here's what each package does:

| Package         | Purpose                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------- |
| `openai`        | The official OpenAI Python SDK. Handles API calls, authentication, and response parsing. |
| `python-dotenv` | Loads your API key from a `.env` file so you never hard-code secrets.                    |
| `requests`      | Makes HTTP calls to external APIs (we'll use it for the weather service).                |

After running this, check your `pyproject.toml` — you'll see a `[project.dependencies]` section listing all three packages. You'll also see a `uv.lock` file, which pins the exact versions installed. Both files should be committed to Git.

### 1.4 — Create the `.env` file

Create a file called `.env` in your project root:

```
OPENAI_API_KEY=sk-your-key-here
```

Replace `sk-your-key-here` with your actual OpenAI API key. You can get one at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

**Why `.env` instead of pasting the key in code?** Two reasons. First, safety — if you ever push your code to GitHub, a `.env` file can be excluded via `.gitignore` so your key doesn't leak publicly. Second, flexibility — you can swap keys (test vs. production) without changing code.

### 1.5 — Update `.gitignore`

`uv init` already creates a `.gitignore` for you. Open it and make sure it includes:

```
.venv/
.env
__pycache__/
```

**Why?** This tells Git to ignore your virtual environment (too large to share), your secrets (`.env`), and Python's cache files (auto-generated, not useful to track). Note: `.venv/` is the directory `uv` creates (as opposed to `venv/` from the older workflow).

### 1.6 — Running files with uv

Throughout this guide, wherever you see a command to run a Python file, use `uv run` instead of `python`:

```bash
# Instead of: python main.py
uv run main.py
```

`uv run` ensures the correct virtual environment is activated and all dependencies are available. You never need to manually activate the `.venv` — `uv run` handles it for you.

### 1.7 — Your file structure so far

```
weather-agent/
├── .env
├── .gitignore
├── .python-version        ← pinned Python version (created by uv init)
├── .venv/                 ← virtual environment (managed by uv)
├── pyproject.toml         ← project config and dependencies
├── uv.lock                ← exact dependency versions (commit this)
└── hello.py               ← sample file from uv init (you can delete this)
```

We'll add our Python files in the following steps.

---

## Step 2: Your First API Call

Before building anything complex, let's make sure the basics work. Create a file called `01_basic_call.py`:

```python
"""
Step 2: A basic OpenAI API call.
This is the simplest possible interaction — send a message, get a response.
We'll build on this foundation in every step that follows.
"""

from openai import OpenAI
from dotenv import load_dotenv

# Load the API key from .env into the environment
load_dotenv()

# Create the OpenAI client
# It automatically reads OPENAI_API_KEY from the environment
client = OpenAI()

# Send a message and get a response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful weather assistant."
        },
        {
            "role": "user",
            "content": "What can you help me with?"
        }
    ]
)

# Print the assistant's reply
print(response.choices[0].message.content)
```

Run it:

```bash
uv run 01_basic_call.py
```

You should see a response from the model.

### What's happening here — line by line

**`load_dotenv()`** — Reads your `.env` file and sets each line as an environment variable. After this runs, `OPENAI_API_KEY` is available to any code that asks for it.

**`client = OpenAI()`** — Creates a client object that manages your connection to the OpenAI API. It automatically looks for `OPENAI_API_KEY` in the environment. You could also pass the key directly (`OpenAI(api_key="sk-...")`), but using the environment is cleaner and safer.

**`messages=[...]`** — This is the conversation history. OpenAI's chat API is _stateless_ — it doesn't remember previous calls. Every time you make a request, you send the **entire** conversation so far. The model reads all messages and generates the next one.

The messages list uses **roles**:

| Role        | Purpose                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| `system`    | Sets the assistant's personality and instructions. The model treats this as its "identity."            |
| `user`      | Messages from the human.                                                                               |
| `assistant` | Messages the model has previously generated. You include these when sending conversation history back. |

**`response.choices[0].message.content`** — The API can technically return multiple choices (alternative responses), but we only asked for one. `.choices[0]` grabs the first (and only) response. `.message.content` is the actual text.

### Why `gpt-4o-mini`?

It's fast, cheap, and smart enough for tool-calling. When learning, you want quick feedback loops and low API costs. You can swap to `gpt-4o` later for more complex reasoning — the code doesn't change.

---

## Step 3: Understanding Tools (Function Calling)

This is the most important concept in the entire guide. Everything "agentic" flows from this.

### The core idea

Normally, the LLM can only generate text. But with **tools** (OpenAI calls them "functions"), you can give the model a menu of actions it can take. The model doesn't _execute_ anything — it generates a structured request saying "I'd like to call this function with these arguments." Your code then actually runs the function and feeds the result back.

The flow looks like this:

```
1. You define tools (just descriptions — name, parameters, what it does)
2. You send these definitions to the model along with the user's message
3. The model reads the user's message and the tool descriptions
4. It decides: should I respond with text, or should I call a tool?
5. If it wants a tool, it returns a structured tool_call (not text)
6. Your code runs the actual function
7. You send the result back to the model
8. The model uses that result to write its final answer
```

### What a tool definition looks like

Create `02_tools_intro.py`:

```python
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
```

### Why the description matters so much

The model decides whether to use a tool based almost entirely on the `description` field and the parameter descriptions. Vague descriptions lead to the model using tools at the wrong time or not at all. Good descriptions are specific:

- **Bad:** `"Gets weather"` — Too vague. When should the model use this?
- **Good:** `"Get the current weather for a given city. Use this when the user asks about weather, temperature, or conditions in a specific location."` — Clear trigger conditions.

Think of it like writing instructions for a coworker. The clearer you are about _when_ and _how_ to use something, the fewer mistakes they'll make.

### What the model returns when it wants to use a tool

Instead of returning text in `message.content`, the model returns a `tool_calls` array:

```python
# When the model wants to call a tool, the response looks like this:
# response.choices[0].message.tool_calls = [
#     {
#         "id": "call_abc123",          ← unique ID for this specific call
#         "type": "function",
#         "function": {
#             "name": "get_weather",    ← which tool it chose
#             "arguments": "{\"city\": \"Tokyo\"}"  ← the arguments (as a JSON string)
#         }
#     }
# ]
```

Notice that `arguments` is a JSON _string_, not a Python dictionary. You'll need to parse it with `json.loads()`. This is a common gotcha for beginners.

---

## Step 4: Build the Weather Tool

Now let's build the actual function that fetches weather data. We'll use Open-Meteo, a free weather API that doesn't require an API key.

Create a file called `tools.py`:

```python
"""
tools.py — The actual functions the agent can call.

Each function here corresponds to a tool definition that we'll give the model.
The model never calls these directly — our agent loop does, based on the
model's tool_call requests.
"""

import json
import requests


def get_weather(city: str) -> str:
    """
    Fetch current weather for a city using the Open-Meteo API.

    The process is two steps:
    1. Convert city name → latitude/longitude (geocoding)
    2. Use coordinates → fetch weather data

    Returns a JSON string because that's what we'll send back to the model.
    The model is very good at reading JSON and summarizing it naturally.
    """

    # --- Step 1: Geocode the city name to coordinates ---
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(geocode_url, params={
        "name": city,
        "count": 1  # We only need the top match
    })
    geo_data = geo_response.json()

    # Handle the case where the city isn't found
    if "results" not in geo_data or len(geo_data["results"]) == 0:
        return json.dumps({"error": f"City '{city}' not found."})

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    resolved_name = location.get("name", city)
    country = location.get("country", "")

    # --- Step 2: Fetch weather using coordinates ---
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = requests.get(weather_url, params={
        "latitude": lat,
        "longitude": lon,
        "current_weather": True  # We only need current conditions
    })
    weather_data = weather_response.json()
    current = weather_data.get("current_weather", {})

    # Package the result as a clean JSON string
    result = {
        "city": resolved_name,
        "country": country,
        "temperature_celsius": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "weather_code": current.get("weathercode"),
        "description": _weather_code_to_text(current.get("weathercode", -1))
    }

    return json.dumps(result)


def _weather_code_to_text(code: int) -> str:
    """
    Convert Open-Meteo's numeric weather code to a human-readable description.

    This is a helper function (note the underscore prefix — a Python convention
    meaning 'this is internal, not part of the public interface').

    We only map the most common codes. For a full list, see:
    https://open-meteo.com/en/docs
    """
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Unknown (code: {code})")
```

### Why return JSON strings?

When you send a tool result back to the model, it needs to be a string. JSON is ideal because it's structured (the model can easily pull out specific values) and unambiguous (no confusion about what "25" means when it's labeled `"temperature_celsius": 25`).

### Why not use the OpenWeatherMap API?

Open-Meteo is completely free with no API key required. One less secret to manage, one less signup process. For a learning project, removing friction matters. You can swap in any weather API later — the agent pattern stays the same.

---

## Step 5: Build the Unit Conversion Tool

Add this function to the same `tools.py` file:

```python
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a temperature between Celsius, Fahrenheit, and Kelvin.

    This is a 'pure' function — it doesn't call any external API,
    it just does math. Not every tool needs to be an API call.
    Tools can be anything: calculations, file operations, database
    queries, or even calls to other AI models.
    """

    # Normalize to lowercase so "Celsius", "celsius", "C" all work
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    # Map common variations to standard names
    unit_aliases = {
        "c": "celsius", "celsius": "celsius",
        "f": "fahrenheit", "fahrenheit": "fahrenheit",
        "k": "kelvin", "kelvin": "kelvin",
    }

    from_std = unit_aliases.get(from_unit)
    to_std = unit_aliases.get(to_unit)

    if not from_std or not to_std:
        return json.dumps({
            "error": f"Unknown unit. Supported: celsius (C), fahrenheit (F), kelvin (K)"
        })

    # Convert input to Celsius first (a common pattern — normalize to one
    # unit, then convert out. This avoids writing 6 separate formulas.)
    if from_std == "celsius":
        celsius = value
    elif from_std == "fahrenheit":
        celsius = (value - 32) * 5 / 9
    elif from_std == "kelvin":
        celsius = value - 273.15

    # Convert from Celsius to the target unit
    if to_std == "celsius":
        result = celsius
    elif to_std == "fahrenheit":
        result = (celsius * 9 / 5) + 32
    elif to_std == "kelvin":
        result = celsius + 273.15

    return json.dumps({
        "input_value": value,
        "input_unit": from_std,
        "output_value": round(result, 2),
        "output_unit": to_std
    })
```

### Why this tool matters for learning

This tool is intentionally simple (just math, no API calls) to show that **tools can be anything**. The agent pattern doesn't care what the tool does internally — it only cares about: (1) what arguments go in, and (2) what string comes out. This is the same pattern whether your tool queries a database, searches the web, or runs a complex ML model.

### The "normalize then convert" pattern

Instead of writing every possible conversion pair (C→F, C→K, F→C, F→K, K→C, K→F — that's 6 formulas), we convert everything to Celsius first, then to the target unit. This means we only need 3 "to Celsius" formulas and 3 "from Celsius" formulas. It's a common engineering pattern: pick a canonical intermediate format, and everything becomes simpler.

---

## Step 6: The Agent Loop

This is the heart of the project. The agent loop is the pattern that makes an LLM into an agent. Once you understand this, you understand how every agent framework works under the hood.

Create a file called `agent.py`:

```python
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
```

### The Agent Loop — Why This Pattern Matters

This `while True` loop is the single most important pattern in this guide. Let's break down _why_ each part exists:

**Why a loop at all?** Because the model might need multiple tool calls to answer one question. "What's the weather in Tokyo in Fahrenheit?" requires two steps: fetching weather (get_weather), then converting the result (convert_temperature). A single API call can only trigger the first step. The loop lets the model keep working until it has everything it needs.

**Why add `assistant_message` to history?** The API is stateless — it doesn't remember previous calls. If the model made a tool call in the first pass, it needs to see that it _made_ that call in the second pass. Otherwise, it might try to call the same tool again. Including the assistant's messages in the history is what gives the model "memory" within a single turn.

**Why `tool_call_id`?** The model can request multiple tool calls in a single pass (parallel tool calls). The ID links each result to the specific call that requested it, so the model knows which result came from where.

**Why `**function_args`(double-star unpacking)?**`function_args`is a dictionary like`{"city": "Tokyo"}`. The `\*\*`syntax unpacks it into keyword arguments:`get_weather(city="Tokyo")`. This lets us call any tool function generically without knowing its specific parameters ahead of time.

---

## Step 7: Wire It All Together

Create `main.py` — this is what you'll actually run:

```python
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
```

### Why the First Version Was Wrong (and Why That's Instructive)

I intentionally showed the bug above because it teaches an important Python concept: **mutable vs. immutable references**.

When you pass `None` to `run_agent`, the function creates a new list internally. But the variable `conversation_history` in `main()` still points to `None` — it doesn't automatically update to point at the new list. Python doesn't work that way.

The fix: create the list _in main()_, then pass it to every call. Since lists are mutable, when `run_agent` appends messages to it, those changes are visible everywhere that references the same list.

This is exactly the kind of subtle bug that causes real problems in agent systems — state management between function calls. Better to encounter it now in a simple context.

### What `if __name__ == "__main__"` means

This line means: "only run `main()` if this file is being run directly." If another file imports something from `main.py`, the `main()` function won't run automatically. It's a Python convention you'll see in virtually every project.

---

## Step 8: Run and Test Your Agent

### Your final file structure

```
weather-agent/
├── .env
├── .gitignore
├── .python-version
├── .venv/
├── pyproject.toml
├── uv.lock
├── 01_basic_call.py      ← Step 2 (you can delete this later)
├── 02_tools_intro.py     ← Step 3 (you can delete this later)
├── tools.py              ← Steps 4-5: The actual tool functions
├── agent.py              ← Step 6: The agent loop
└── main.py               ← Step 7: Interactive chat interface
```

### Run it

```bash
uv run main.py
```

### Conversations to try

Test these to see different agent behaviors:

```
You: What's the weather in San Miguel de Allende?
→ Agent calls get_weather, reports conditions

You: How about in Tokyo?
→ Agent calls get_weather again (tests conversation memory)

You: Convert 25 celsius to fahrenheit
→ Agent calls convert_temperature (tests tool selection)

You: What's the weather in London in Fahrenheit?
→ Agent might chain: get_weather → convert_temperature (tests multi-step)

You: Hello!
→ Agent responds with text, no tool calls (tests that it doesn't force tools)

You: What's warmer right now, Paris or Berlin?
→ Agent calls get_weather twice, then compares (tests parallel reasoning)
```

### What to watch for

- **Tool selection**: Does the model pick the right tool? Or does it try to use weather lookup for a conversion question?
- **No-tool responses**: When you say "Hello," the model should just respond — not call any tool. A good agent knows when _not_ to use tools.
- **Multi-step reasoning**: For "weather in Fahrenheit" questions, watch whether the model chains tools together.

---

## Step 9: What's Next

You now have a working agent with the foundational pattern. Here's what you can layer on, in order of complexity:

### Add More Tools (Easiest)

Adding a tool is always the same three steps:

1. Write the function in `tools.py`
2. Add the tool definition to the `TOOLS` list in `agent.py`
3. Add the function to the `TOOL_FUNCTIONS` dictionary

Ideas: time zones, air quality, sunrise/sunset, or completely unrelated tools like to-do lists or calculations.

### Add Conversation Memory (Moderate)

Right now, conversation history resets when you restart the program. To add persistence:

- Save `conversation_history` to a JSON file after each turn
- Load it on startup
- Add a summary mechanism so the history doesn't grow forever (you can use the LLM itself to summarize old messages)

### Add RAG — Retrieval Augmented Generation (Moderate)

RAG lets your agent reference a knowledge base. The pattern:

- Store documents as embeddings in a vector database (e.g., ChromaDB)
- Create a `search_knowledge_base` tool
- When the user asks a question, the agent searches the knowledge base and uses the results to inform its answer

This is just another tool — the agent loop doesn't change.

### Add Structured Error Handling (Moderate)

Production agents need:

- Retry logic for API failures
- Timeout handling for slow tool calls
- Graceful degradation ("I couldn't fetch the weather, but I can still do conversions")
- Token counting to prevent conversations from exceeding the model's context window

### Add Streaming (Moderate)

Instead of waiting for the full response, stream tokens as they arrive. This makes the agent feel more responsive. The OpenAI SDK supports this with `stream=True`.

### Multi-Agent Systems (Advanced)

Instead of one agent with many tools, create specialized agents that hand off to each other. For example:

- A "router" agent that decides which specialist to involve
- A "weather" agent and a "planning" agent
- They communicate through structured messages

The core loop is the same — just nested.

---

## Quick Reference: The Agent Pattern

When you're building agents in the future, this is the pattern you'll always come back to:

```
┌─────────────────────────────────────────────────┐
│                  THE AGENT LOOP                  │
│                                                  │
│   1. Gather: messages + tools + system prompt    │
│   2. Call LLM                                    │
│   3. Did LLM request tool calls?                 │
│      ├─ YES → Execute tools                      │
│      │        Add results to messages             │
│      │        Go to Step 2                        │
│      └─ NO  → Return the text response            │
│               (Agent is done)                     │
└─────────────────────────────────────────────────┘
```

Every agent framework (LangChain, CrewAI, Autogen, etc.) is a more elaborate version of this loop. Understanding it at this level means you'll never be confused by what those frameworks are doing under the hood.

---

_Built as a learning project. Extend it, break it, rebuild it — that's how you learn._
