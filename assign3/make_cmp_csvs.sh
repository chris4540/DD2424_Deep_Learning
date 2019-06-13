#!/bin/bash
for init_sig in 1e-1 1e-3 1e-4; do

    echo "With batchnorm" > bn.txt
    grep "Loss" log_bn_${init_sig}.txt | awk '{print $6}' >> bn.txt

    echo "Without batchnorm" > no_bn.txt
    grep "Loss" log_no_bn_${init_sig}.txt | awk '{print $6}' >> no_bn.txt

    paste -d "," bn.txt no_bn.txt > ${init_sig}.csv

    # remove temp files
    rm bn.txt
    rm no_bn.txt
done

