EXT_SUFFIX := $(shell python-config --extension-suffix)
RUNTEST=python -m unittest -v
TESTCASES=tests/*.py

default: ann_f$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	f2py $< -m $*  -h $*.pyf --overwrite-signature
	gfortran -O3 -c -fPIC -ftree-vectorize $< -o $*.o
	f2py -c $*.pyf -I. --opt="-O3 -ftree-vectorize" $*.o

clean:
	rm *.so

test:
	${RUNTEST} ${TESTCASES}
