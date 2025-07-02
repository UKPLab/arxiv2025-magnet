import argparse
import json
import multiprocessing
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from langchain.prompts import PromptTemplate
from vllm import LLM
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import openai

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

print(torch.cuda.get_device_capability()[0])

# Use your local vLLM server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any value works
)

def generate(prompt: str) -> str:
    response = client.completions.create(
    model="model_name",
    prompt=prompt,
    temperature=0.7,
    max_tokens=512
    )
    return response.choices[0].text

class Agent(ABC):
    def __init__(self):
        self.prompt_template = None

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def load_prompt(self, file_name):
        base_dir = "../prompts/"
        file_path = base_dir + file_name
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


class ClientAgent(Agent):
    def __init__(self, example):
        super().__init__()
        self.example = example
        prompt_text = self.load_prompt(f"agent_client.txt")
        self.attitude = (
            f"{self.example['AI_client']['attitude']}: "
            f"{self.example['AI_client']['attitude_instruction']}")
        self.prompt_template = PromptTemplate(
            input_variables=["intake_form", "attitude", "history"],
            template=prompt_text)

    def generate(self, history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(
            intake_form=self.example,
            attitude=self.attitude,
            history=history_text
        )

        return generate(prompt)

class CounselorAgent(Agent):
    def __init__(self):
        super().__init__()
        prompt_text = self.load_prompt(f"agent_psych8k.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        history = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history)
        return generate(prompt)

class Psych8kCounselorAgent(CounselorAgent):
    def __init__(self):
        super().__init__()
        prompt_text = self.load_prompt(f"agent_psych8k.txt")
        self.prompt_template = PromptTemplate(
            input_variables=["history"],
            template=prompt_text)

    def generate(self, history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        prompt = self.prompt_template.format(history=history_text)
        response = generate(prompt)
        response = response.replace('Output:', '')
        response = response.replace('Counselor:', '')
        response = response.strip()
        return response

class TherapySession:
    def __init__(self, example, max_turns):
        self.example = example
        self.client_agent = ClientAgent(example=example)
        self.counselor_agent = Psych8kCounselorAgent()
        self.history = []
        self.max_turns = max_turns

    def _add_to_history(self, role, message):
        self.history.append({"role": role, "message": message})

    def _initialize_session(self):
        example_cbt = self.example['AI_counselor']['CBT']
        self._add_to_history("counselor",example_cbt['init_history_counselor'])
        self._add_to_history("client", example_cbt['init_history_client'])

    def _exchange_statements(self):

        for turn in range(self.max_turns):
            counselor_statement = self.counselor_agent.generate(self.history)
            counselor_statement = counselor_statement.replace('Counselor: ',
                                                              '')
            self._add_to_history("counselor", counselor_statement)

            client_statement = self.client_agent.generate(self.history)
            client_statement = client_statement.replace('Client: ', '')

            self._add_to_history("client", client_statement)

            if '[/END]' in client_statement:
                self.history[-1]['message'] = self.history[-1][
                    'message'].replace('[/END]', '')
                break

    def run_session(self):
        self._initialize_session()
        self._exchange_statements()
        return {
            "example": self.example,
            "history": self.history
        }


def run_therapy_session(index, example, output_dir, total, max_turns):
    output_dir = Path(output_dir)
    file_number = index + 1

    try:
        print(f"Generating example {file_number} out of {total}")
    
        therapy_session = TherapySession(
            example,
            max_turns,
        )
        session_data = therapy_session.run_session()
    
        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name
    
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        error_file_name = f"error_{file_number}.txt"
        error_file_path = output_dir / error_file_name
        tb = e.__traceback__
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(type(e), e, tb)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run therapy sessions in parallel.")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to save the session results.")
    parser.add_argument("-num_pr","--num_processes", type=int, default=None,
                        help="Number of processes to use in the pool."
                             " Defaults to the number of CPU cores "
                             "if not specified.")
    parser.add_argument("-m_turns","--max_turns", type=int, default=20,
                        help="Maximum number of turns for the session.")

    args = parser.parse_args()

    with open("../dataset/evaluation.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(data)
    args_list = [(index, example, output_dir, total, args.max_turns)
                 for index, example in enumerate(data)]

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        for i, _ in enumerate(pool.starmap(run_therapy_session, args_list)):
            print(f"Generating example {i} out of {total}")
