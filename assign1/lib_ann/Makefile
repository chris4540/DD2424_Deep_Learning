EXT_SUFFIX := $(shell python-config --extension-suffix)

default: ann_fortran$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	f2py -c -m $* $<

clean:
	rm ann_fortran$(EXT_SUFFIX)
