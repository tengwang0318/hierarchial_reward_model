Just download PRM800K dataset, create another folder named prm_dataset and place these 4 jsonl files into that folder.

In this project, we only use phase1 part which contains manual annotation.

construct_dataset folder is used for constructing training dataset when handling with manual annotation data, and this process will create a new folder named phase1 that contains ORM, PRM and HRM training data.

In auto-annotation process, we only use the question and ground truth from PRM800K dataset. Self-training module will generate the reasoning process and label it autonomously.
