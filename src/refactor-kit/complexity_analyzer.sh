#!/bin/bash
# This script uses the radon library to analyze cyclomatic complexity of Python
# source files in a specified directory. It computes and prints complexity metrics
# for each function, assigning grades from A to F based on complexity scores.

# The format is: <Type> <Line>:<Column> <Name> - <Grade> (<CC Score>)

# F = Function
# M = Method
# C = Class

radon cc ./ -s
