"""
@Time    : 2023/5/18 00:40
@File    : token_counter.py
@From    : https://github.com/geekan/MetaGPT/blob/main/metagpt/utils/token_counter.py
ref1: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
ref2: https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/llm/token_counter.py
ref3: https://github.com/hwchase17/langchain/blob/master/langchain/chat_models/openai.py
"""
from typing import NamedTuple
import tiktoken
import abc


class Singleton(abc.ABCMeta, type):
    """
    Singleton metaclass for ensuring only one instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Call method for the singleton metaclass."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


TOKEN_COSTS = {
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0301": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-0613": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    "gpt-3.5-turbo-16k-0613": {"prompt": 0.003, "completion": 0.004},
    "gpt-4-0314": {"prompt": 0.03, "completion": 0.06},
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-32k-0314": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-0613": {"prompt": 0.06, "completion": 0.12},
    "text-embedding-ada-002": {"prompt": 0.0004, "completion": 0.0},
    "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "deepseek-reasoner": {"prompt": 0, "completion": 0},
}


def count_message_tokens(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_message_tokens(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_message_tokens(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def count_string_tokens(string: str, model_name: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
        string (str): The text string.
        model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float


class CostManager(metaclass=Singleton):

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.total_call_count = 0

        self.total_kg_prompt_tokens = 0
        self.total_kg_expert_prompt_tokens = 0
        self.total_discuss_prompt_tokens = 0
        self.total_decision_prompt_tokens = 0
        self.total_recruit_tokens = 0
        self.initial_assessment_prompt_tokens = 0

        self.total_kg_completion_tokens = 0
        self.total_kg_expert_completion_tokens = 0
        self.total_discuss_completion_tokens = 0
        self.total_decision_completion_tokens = 0
        self.total_recruit_completion_tokens = 0
        self.initial_assessment_completion_tokens = 0

    def update_sub_cost(self, prompt_tokens, completion_tokens, model, action):

        if "recruit" in action and "gap" not in action:
            self.total_recruit_tokens += prompt_tokens
            self.total_recruit_completion_tokens += completion_tokens
        elif "ask_gap" in action:
            self.total_kg_prompt_tokens += prompt_tokens
            self.total_kg_completion_tokens += completion_tokens
        elif (
            "gap_recruit" in action
            or "gap_recruited" in action
            or "gap_identified" in action
        ):
            self.total_kg_expert_prompt_tokens += prompt_tokens
            self.total_kg_expert_completion_tokens += completion_tokens
        elif "discuss" in action:
            self.total_discuss_prompt_tokens += prompt_tokens
            self.total_discuss_completion_tokens += completion_tokens
        elif action == "decision":
            self.total_decision_prompt_tokens += prompt_tokens
            self.total_decision_completion_tokens += completion_tokens
        elif "initial_assessment" in action:
            self.initial_assessment_prompt_tokens += prompt_tokens
            self.initial_assessment_completion_tokens += completion_tokens
        print(
            f"--- Sub-action cumulative tokens ---\n"
            f"Total recruit: Prompt {self.total_recruit_tokens}, Completion {self.total_recruit_completion_tokens}\n"
            f"Total initial assessment: Prompt: {self.initial_assessment_prompt_tokens}, Completion {self.initial_assessment_completion_tokens}\n"
            f"Total KG: Prompt {self.total_kg_prompt_tokens}, Completion {self.total_kg_completion_tokens}\n"
            f"Total KG Expert: Prompt {self.total_kg_expert_prompt_tokens}, Completion {self.total_kg_expert_completion_tokens}\n"
            f"Total Discuss: Prompt {self.total_discuss_prompt_tokens}, Completion {self.total_discuss_completion_tokens}\n"
            f"Total Decision: Prompt {self.total_decision_prompt_tokens}, Completion {self.total_decision_completion_tokens}\n"
        )

    def update_cost(self, prompt_tokens, completion_tokens, model, action):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        cost = (
            prompt_tokens * TOKEN_COSTS[model]["prompt"]
            + completion_tokens * TOKEN_COSTS[model]["completion"]
        ) / 1000
        self.total_cost += cost
        self.total_call_count += 1

        self.update_sub_cost(prompt_tokens, completion_tokens, model, action)

        print(
            f"Total running cost: ${self.total_cost} | Total API calls: {self.total_call_count} | "  # | Max budget: ${CONFIG.max_budget:.3f} |
            f"Total prompt_tokens: {self.total_prompt_tokens} | Total completion_tokens {self.total_completion_tokens} | "
            f"Current cost: ${cost}, {prompt_tokens=}, {completion_tokens=}"
        )
        # CONFIG.total_cost = self.total_cost

    def get_costs(self) -> Costs:
        return Costs(
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_cost,
            self.total_budget,
        )


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    model="gpt-4"
    prompt_token = count_message_tokens(messages, model)
    response_token = count_string_tokens("Fine", model)
    cost = CostManager()
    cost.update_cost(prompt_token, response_token, model=model)
    print(cost.get_costs())
