#!/bin/bash
# For assignment 4 basic part

echo "iteration,smooth_cost" > basic_cost.csv
grep iteration rnn.log | awk '{print $2 "," $4}' >> basic_cost.csv

python plt_csv.py basic_cost.csv
