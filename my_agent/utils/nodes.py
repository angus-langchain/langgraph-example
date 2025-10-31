import random
import nltk
from langsmith import trace, traceable
from nltk.corpus import words

# Download words if not already available
try:
    word_list = words.words()
except LookupError:
    nltk.download("words")
    word_list = words.words()

# Different data sizes to simulate varying payload sizes
data_sizes = [
    100,
    150,
    180,
    200,
    210,
    230,
    250,
    300,
    500,
    800,
    1200,
    1800,
    2352,
    2800,
    3500,
    5000,
    8000,
    10000,
]

ITERATIONS = 2

# Pre-generate data arrays with random words
data_array = [" ".join(random.choices(word_list, k=(size // 5))) for size in data_sizes]


@traceable(name="method_1", run_type="chain")
def another_trace_method(messages):
    """Traceable method 1."""
    one_more_trace_method(messages)


@traceable(name="method_2", run_type="llm")
def one_more_trace_method(messages):
    """Traceable method 2."""
    pass


system_prompt = """Be a helpful assistant"""


def should_continue(state):
    """Determine whether to continue or end the graph execution."""
    messages = state["messages"]

    # Randomly decide whether to continue or end (25% chance to end)
    if random.random() < 0.25:
        return "end"
    else:
        return "continue"


def call_model(state, config):
    """Simulate calling a model with random data."""
    input_data = random.choice(data_array)
    output_data = random.choice(data_array)

    messages = state["messages"]
    messages = [
        {"role": "system", "content": system_prompt, "data": input_data}
    ] + messages

    # Simulate model response
    random_words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    simulated_content = " ".join(random.choices(random_words, k=10))
    response = {"role": "assistant", "content": simulated_content, "data": output_data}

    # Call traceable methods
    another_trace_method(messages)
    another_trace_method(response)

    # Remove data field before returning
    del response["data"]

    return {"messages": [response]}


def call_tool_node(state, config):
    """Simulate calling a tool with random data."""
    input_data = random.choice(data_array)
    output_data = random.choice(data_array)

    # Simulate tool execution with fake response
    fake_tool_response = {
        "role": "tool",
        "content": f"Tool executed successfully. Input: {input_data[:100]}... Output: {output_data[:100]}...",
        "tool_call_id": "123",
        "name": "tavily_search_results_json",
    }

    # Create multiple traces with different model tags
    for _ in range(ITERATIONS):
        with trace(
            "MethodTrace",
            inputs={"input": input_data},
            tags=[random.choice(["model:claude", "model:gpt-4o", "model:gpt-4o-mini"])],
        ):
            another_trace_method(fake_tool_response)
            another_trace_method(fake_tool_response)

    return {"messages": [fake_tool_response]}
