#!/bin/bash

# Make for 3-layer
# batchnorm
echo "with_batchnorm" > bn.txt
grep "Loss" bn_3l_log.txt | awk '{print $6}' >> bn.txt

# w/o batchnorm
echo "wo_batchnorm" > no_bn.txt
grep "Loss" no_bn_3l_log.txt | awk '{print $6}' >>  no_bn.txt

paste -d "," bn.txt no_bn.txt > loss_3l.csv

rm bn.txt no_bn.txt