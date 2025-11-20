# Fall detection transfer learning based on Transformer and CNN

Platform: Python 3.9

Data: The wrist data of five volunteers were collected by the self-made hardware based on STM32F103 C8T6. It includes three daily activities (standing, walking, and quickly sitting down) and two fall behaviors (getting up after falling and not getting up after falling). 

Main.py for balanced data pre-training

预训练.py for original data pre-training (Public and large datasets: Up-fall dataset)

The two files have similar function, just change the input data.

微调.py for self-built dataset, which aims to learn the true distribution of actual data and solve the out-of-distribution problem.

functions2.py for necessary functions 

The corresponding paper of the code: to be published.
