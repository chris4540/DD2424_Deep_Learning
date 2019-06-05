#!/bin/bash
# For assignment 4 basic part

echo "iteration,smooth_cost" > basic_cost.csv
grep iteration rnn_log.txt | awk '{print $2 "," $4}' >> basic_cost.csv

# python plt_csv.py basic_cost.csv
echo "iteration,smooth_cost" > trump_cost.csv
grep iteration trump_tweet_log.txt | awk '{print $2 "," $4}' >> trump_cost.csv


