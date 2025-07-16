## PANAS Folders/Files

- prompts: Contains the prompts used in evaluating the ratings for emotions before and after the counseling session.
- cost.ipynb: Calculates the GPT-4o costs involved in the evaluation.
- rating.ipynb: Calculates the average change in score for positive and negative emotions over all sessions generated using a particular method for clients with positive, neutral and negative attitudes.
- panas_before-gpt4o.py: Used to evaluate the scores for each item of positive and negative emotions of clients before counseling. It contains the following arguments:
  - ```-o```: This argument takes the path of the output directory where the PANAS scores before counseling will be stored.
  - ```-m_iter```: Number of times GPT-4o is run to score a particular item for a session. For multiple runs the average score is noted.

To run this script use ```python panas_before-gpt4o.py -o "output_dir" -m_iter num```

- panas_after-gpt4o.py: Used to evaluate the scores for each item of positive and negative emotions of clients after counseling. It contains the following arguments:
  - ```-i```: This argument takes the path of the directory containing the counseling sessions of the clients whose emotions are to be rated.
  - ```-o```: This argument takes the path of the output directory where the PANAS scores after counseling will be stored.
  - ```-m_iter```: Number of times GPT-4o is run to score a particular item for a session. For multiple runs the average score is noted.

To run this script use ```python panas_after-gpt4o.py -i "input_dir" -o "output_dir" -m_iter num```
