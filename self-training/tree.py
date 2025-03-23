"""
The data structure for MCTS
"""
from grading.grader import grade_answer
from envs import *


class Node:
    def __init__(self, parent=None, height=0, question="", number_of_children=NUMBER_OF_CHILDREN, score=-1):
        self.previous_answer = []
        self.parent = parent
        self.number_of_children = number_of_children
        self.children = []
        self.height = height
        self.question = question
        self.contain_answer = False
        self.answer = None
        self.score = score
        self.is_correct = False
        self.should_stop = False
        self.is_leaf_node = False

    def add_children(self, children_list):
        self.children = children_list
        if len(self.children) != self.number_of_children:
            print(f"question: {self.question} children doesn't equal number of children: {self.number_of_children}")

    def have_the_answer(self, answer, ground_truth):
        self.answer = answer
        self.contain_answer = True
        self.is_correct = grade_answer(answer, ground_truth)

    def set_previous_answer(self, previous_answer):
        self.previous_answer = previous_answer[:]
        assert len(self.previous_answer) == self.height - 1
