#!/bin/bash
# For code submission


cat << EOF > ./submission_code.py
# This file is for submission only. All related files are concatenated to this file for submission.
EOF

echo "# ================================================================================================" >> ./submission_code.py
echo "# ex1_run.py" >> ./submission_code.py
echo "# ================================================================================================" >> ./submission_code.py
cat ./ex1_run.py >> ./submission_code.py


echo "# ================================================================================================" >> ./submission_code.py
echo "# load_batch.py" >> ./submission_code.py
cat ./load_batch.py >> ./submission_code.py


echo "# ================================================================================================" >> ./submission_code.py
echo "# visual.py" >> ./submission_code.py
cat ./visual.py >> ./submission_code.py


echo "# ================================================================================================" >> ./submission_code.py
echo "# one_layer_ann.py" >> ./submission_code.py
cat ./one_layer_ann.py >> ./submission_code.py


echo "# ================================================================================================" >> ./submission_code.py
echo "# tests/test_grad.py" >> ./submission_code.py
cat tests/test_grad.py >> ./submission_code.py