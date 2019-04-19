RUNTEST=nosetests --processes=-1 --process-timeout=100 -v
# RUNTEST=python -m unittest -v

TESTCASES=tests/*.py

test:
	${RUNTEST} ${TESTCASES}

.PHONY: lib_clsr
lib_clsr:
	make -C lib_clsr

clean:
	make -C lib_clsr clean
