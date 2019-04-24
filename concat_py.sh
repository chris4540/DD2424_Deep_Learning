#!/bin/bash
# For code submission


cat << EOF > ./submission_code.py
# This file is for submission only. All related files are concatenated to this file for submission.
EOF

# Assign 1
# files="
# clsr/_base.py
# clsr/one_layer_network.py
# clsr/svm.py
# lib_clsr/ann.py
# lib_clsr/init.py
# lib_clsr/svm.py
# lib_clsr/utils.py
# scripts/cmp_ann_svm.py
# scripts/combine_all_skills.py
# scripts/GridSearchCV_ann.py
# scripts/plt_ex1_basic.py
# scripts/run_1l_ann.py
# scripts/run_svm.py
# tests/test_ann_func.py
# tests/test_svm_func.py
# tests/utils.py
# "

# Assign 2
files="
clsr/two_layer_network.py
lib_clsr/ann.py
lib_clsr/init.py
lib_clsr/utils.py
assign2/run_a2_ex3_f3.py
assign2/run_a2_ex3_f4.py
assign2/run_a2_ex4_best.py
assign2/run_a2_ex4_get_search_val.py
assign2/run_a2_ex4_search_lmbd.py
tests/test_ann_2l.py
tests/test_sanity_2l_ann.py
tests/utils.py
"

for file in ${files}; do
    echo ${file}
    echo "# ================================================================================================" >> ./submission_code.py
    echo "# ${file}" >> ./submission_code.py
    echo "# ================================================================================================" >> ./submission_code.py
    cat ${file} >> ./submission_code.py
done
