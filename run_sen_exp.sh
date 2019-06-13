#!/bin/bash

for is_bn in False;do
# for is_bn in True False;do
    export is_bn
    if [[ ${is_bn} = True ]]; then
        logfile="log_bn"
    else
        logfile="log_no_bn"
    fi
    for init_sig in 1e-1 1e-3 1e-4; do
        export init_sig
        python -u run_sen_init.py | tee ${logfile}_${init_sig}.txt
    done
done

