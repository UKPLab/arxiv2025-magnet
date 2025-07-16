## src Folders/Files

To be compatible with multiprocessing, we host the Llama3-8B-Instruct models used for the agents in a local server using the following:

```
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype float16 \
  --host 0.0.0.0 \
  --port $PORT
```

Here replace "$MODEL_PATH" with the local model path and port by the hosting port. In this codebase we use 8000 in all scripts. If a different port is used please replace it each script in this portion:

```
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any value works
)
```

- inference-parallel-cactus.py: This file is used to generate synthetic counseling sessions using CACTUS. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python inference-parallel-cactus.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- inference-parallel-mascs.py: This file is used to generate synthetic counseling sessions using MASCS-Gen. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python inference-parallel-mascs.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- inference-parallel-psych8k.py: This file is used to generate synthetic counseling sessions using Psych8k. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python inference-parallel-psych8k.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- mascs-parallel-no-cbt-ablation.py: This file is used to generate synthetic counseling sessions using an ablation of MASCS-Gen with no CBT agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python mascs-parallel-no-cbt-ablation.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- mascs-parallel-no-tech-ablation.py: This file is used to generate synthetic counseling sessions using an ablation of MASCS-Gen with no Technique agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python mascs-parallel-no-tech-ablation.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- mascs-parallel-no-cbt-no-tech-ablation.py: This file is used to generate synthetic counseling sessions using an ablation of MASCS-Gen with no CBT and Technique agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python mascs-parallel-no-cbt-no-tech-ablation.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

For using Qlora fine-tuned models we host them on another local server using:

```
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype float16 \
  --host 0.0.0.0 \
  --port $PORT
```

Here replace "$MODEL_PATH" with the local model path for the qlora fine-tuned model (different for CACTUS, MASCS and Psych8k) and port by the hosting port. In this codebase we use 8005 in all scripts. If a different port is used please replace it each script in this portion:

```
counselor = openai.OpenAI(
    base_url="http://localhost:8005/v1",
    api_key="dummy-key"  # Any value works
)
```

- qlora-cactus-parallel.py: This file is used to generate synthetic counseling sessions using the Llama Qlora fine-tuned with the CACTUS data as the counselor agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python qlora-cactus-parallel.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- qlora-mascs-parallel.py: This file is used to generate synthetic counseling sessions using the Llama Qlora fine-tuned with the MASCS-Gen data as the counselor agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python qlora-mascs-parallel.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- qlora-psych8k-parallel.py: This file is used to generate synthetic counseling sessions using the Llama Qlora fine-tuned with the Psych8k data as the counselor agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python qlora-psych8k-parallel.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.
