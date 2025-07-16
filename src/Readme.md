## src Folders/Files

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
To run this script use ```python mascs-parallel-no-cbt-ablation.py.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- mascs-parallel-no-tech-ablation.py: This file is used to generate synthetic counseling sessions using an ablation of MASCS-Gen with no Technique agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python mascs-parallel-no-tech-ablation.py.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

- mascs-parallel-no-cbt-no-tech-ablation.py: This file is used to generate synthetic counseling sessions using an ablation of MASCS-Gen with no CBT and Technique agent. It contains the following arguments:
  - ```-o```: Path to output directory to save the generated sessions.
  - ```-num_pr```: Number of process to use in a pool for multiprocessing.
  - ```-m_turns```: Maximum number of dialogue turns for the counselor and the client each in a generated therapy session.
To run this script use ```python mascs-parallel-no-cbt-no-tech-ablation.py.py -o "output_dir" -num_pr 8 -m_turns 20``` with 8 processes and a maximum of 20 turns.

