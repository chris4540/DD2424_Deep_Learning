EXT_SUFFIX := $(shell python-config --extension-suffix)

default: ann_fortran$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	bash ./compile.sh

clean:
	rm ann_fortran$(EXT_SUFFIX)
