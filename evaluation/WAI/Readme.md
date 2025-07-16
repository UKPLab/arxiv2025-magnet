## WAI Folders/Files

- prompts: Contains the prompts used in evaluating the 12 different Items in WAI.
- cost.ipynb: Calculates the GPT-4o costs involved in the evaluation.
- rating.ipynb: Calculates the average score for each item over all sessions generated using a particular method.
- wai-gpt4o.py: Used to evaluate the scores for each item of each generated synthetic counseling sessions. It contains the following arguments:
  - ```-i```: This argument takes the path of the directory containing the sessions to be rated.
  - ```-o```: This argument takes the path of the output directory where the WAI scores of the sessions will be stored.
  - ```-m_iter```: Number of times GPT-4o is run to score a particular item for a session. For multiple runs the average score is noted.

To run this script use ```python wai-gpt4o.py -i "input_dir" -o "output_dir" -m_iter num```
