from langchain.prompts import PromptTemplate
import os
import json
import argparse
import os
from openai import AzureOpenAI
import tiktoken
from pathlib import Path

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to count tokens
def num_tokens_from_message(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens_per_message = 3
    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message + len(enc.encode(message["content"]))
    
    total_tokens += 3
    return total_tokens

def calculate_cost(input_message, output):
    input_cost = (num_tokens_from_message(input_message)*(5))/1000000
    output_cost = (num_tokens_from_string(output)*(20))/1000000
    return input_cost+output_cost

def generate_history(history):
    history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
    )

    return history_text

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CTRS results")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Number of times GPT-4o is run for scoring a single session.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    endpoint = "azure endpoint"
    model_name = "gpt-4o"
    deployment = "gpt-4o"
    
    subscription_key = "subscription key"
    api_version = "api version"
    
    evaluator = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    ctrs_list = ["general_1_understanding", "general_2_interpersonal_effectiveness", "general_3_collaboration", "CBT_1_guided_discovery", "CBT_2_focus", "CBT_3_strategy"]
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            score_dict = {}
            total_cost = 0
            for i in range(len(ctrs_list)):
                score = 0
                prompt_text = load_prompt(ctrs_list[i] + ".txt")
                prompt_template = PromptTemplate(
                                    input_variables=["conversation"],
                                    template=prompt_text)
                prompt = prompt_template.format(
                                                    conversation = generate_history(json_data["history"])
                                                )
                messages = [{'role': 'user', 'content': prompt}]
                response = evaluator.chat.completions.create(
                            messages=messages,
                            temperature=0,
                            model=deployment,
                            n=args.max_iter
                            )
                for j in range(args.max_iter):
                    total_cost = total_cost+calculate_cost(messages,response.choices[j].message.content)
                    txt_response = response.choices[j].message.content
                    score = score + int(txt_response.split(',')[0])
                avg_score = score/args.max_iter
                score_dict[ctrs_list[i]] = avg_score
            score_dict["cost"] = total_cost
            save_as_json(score_dict, os.path.join(args.output_dir, filename))
