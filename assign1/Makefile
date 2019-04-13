RUNTEST=python -m unittest -v

TESTCASES=tests/*.py

default:lib_clsr

.PHONY: lib_clsr

lib_clsr:
	make -C lib_clsr

clean:
	make -C lib_clsr clean

test:
	${RUNTEST} ${TESTCASES}
