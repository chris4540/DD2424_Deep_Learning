RUNTEST=python -m unittest -v

TESTCASES=tests/*.py

ann_fortran:
	make -C lib_ann

clean:
	make -C lib_ann clean

test:
	${RUNTEST} tests/test_ann_fortran.py
	# ${RUNTEST} ${TESTCASES}
