EXT_SUFFIX := $(shell python-config --extension-suffix)

default: ann_f$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	f2py $< -m $*  -h $*.pyf --overwrite-signature
	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-blas --link-atlas $<
# 	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-openblas $<
# 	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-blas_mkl $<
#	f2py -c $*.pyf --opt="-O3 -fexternal-blas" --link-atlas $<

clean:
	rm -f *.pyf
	rm -f *.so
	rm -f *.o
	rm -f *.mod
	rm -rf *.pyd
