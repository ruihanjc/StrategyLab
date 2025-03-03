#!/bin/python3

import math
import os
import random
import re
import sys


# Write the implementations of the Student and Result classes.
# The following are output string formats for cut and paste:
# Added student: {name} to the roll of class: {sClass}
# {name} has been promoted to class: {studentClass +1}
# {name} has been retained in class: {studentClass}
# {name} obtained {marks} marks in subject1
# {name} obtained {marks} marks in subject2
# {name} obtained {marks} marks in subject3
# {name} has ordered a recheck in {subject}
# Following is the new result: {name} has been promoted  to class: {studentClass + 1}
# Following is the new result: {name} has been retained in class: {studentClass}

class Student:
    def __init__(self, name, s_class):
        self.name = name
        self.s_class = s_class
        print("Added student: {} to the roll of class: {}".format(name, s_class))

    def get_name(self):
        return self.name

    def publish(self):
        if self.result >= 33.33:
            return "{} has been promoted to class: {}".format(self.name, self.s_class + 1)
        else:
            return "{} has been retained in class: {}".format(self.name, self.s_class)


class Result(Student):
    def __init__(self, subject1, subject2, subject3, name, s_class):
        super().__init__(name, s_class)
        self.subject1 = subject1
        self.subject2 = subject2
        self.subject3 = subject3
        self.result = 0
        print("{} obtained {} marks in subject1".format(self.name, subject1))
        print("{} obtained {} marks in subject2".format(self.name, subject2))
        print("{} obtained {} marks in subject3".format(self.name, subject3))

    def calculate_result(self):
        self.result = (self.subject1 + self.subject2 + self.subject3) * 100 / 300
        print(self.publish(), end="")
        return ""

    def change_marks(self, new_marks, subject):
        print("{} has ordered a recheck in {}".format(self.name, subject))

        if subject == "subject1":
            self.subject1 = new_marks
        elif subject == "subject2":
            self.subject2 = new_marks
        elif subject == "subject3":
            self.subject3 = new_marks

        s = self.publish()
        print("Following is the new result: {}".format(s))

        return ""


if __name__ == '__main__':
    names = input().split()
    marks = []

    for i in range(len(names)):
        temp = list(map(int, input().split()))
        marks.append(temp)

    cla = list(map(int, input().split()))

    r1 = Result(marks[0][0], marks[0][1], marks[0][2], names[0], cla[0])
    r2 = Result(marks[1][0], marks[1][1], marks[1][2], names[1], cla[1])
    r3 = Result(marks[2][0], marks[2][1], marks[2][2], names[2], cla[2])
    r4 = Result(marks[3][0], marks[3][1], marks[3][2], names[3], cla[3])
    r5 = Result(marks[4][0], marks[4][1], marks[4][2], names[4], cla[4])

    sub = input()
    new_marks = int(input())

    print(r1.calculate_result())
    print(r2.calculate_result())
    print(r3.calculate_result())
    print(r4.calculate_result())
    print(r5.calculate_result())

    print(r1.change_marks(new_marks, sub))
    print(r3.change_marks(new_marks, sub))
    print(r5.change_marks(new_marks, sub))