## CTRS Folders/Files

- prompts: Contains the prompts used in evaluating the six different categories of CTRS.
- cost.ipynb: Calculates the GPT-4o costs involved in the evaluation.
- rating.ipynb: Calculates the average score for each criteria over all sessions generated using a particular method.
- ctrs-gpt4o.py: Used to evaluate the scores for each criteria of each generated synthetic counseling sessions. It contains the following arguments:
  - ```-i```: This argument takes the path of the directory containing the sessions to be rated.
  - ```-o```: This argument takes the path of the output directory where the CTRS scores of the sessions will be stored.
  - ```-m_iter```: Number of times GPT-4o is run to score a session. For multiple runs the average score is noted.

To run this script use ```python ctrs-gpt4o.py -i "input_dir" -o "output_dir" -m_iter num```
