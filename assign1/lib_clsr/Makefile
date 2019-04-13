EXT_SUFFIX := $(shell python-config --extension-suffix)
RUNTEST=python -m unittest -v
TESTCASES=tests/*.py

default: ann_f$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	f2py $< -m $*  -h $*.pyf --overwrite-signature
	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-openblas $<
# 	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-blas $<
# 	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-blas_mkl $<

clean:
	rm *.so

test:
	${RUNTEST} ${TESTCASES}
