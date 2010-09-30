
RedHat 64bit
============

(Written Sep'10)

For making pytables install correctly
-------------------------------------

Install cython: ::

	$ pip install cython

Download and install a binary distribution of HDF5. ::

	$ wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.5-patch1.tar.gz
	$ ./configure --prefix=<dir>
	$ make
	$ make install
	
Also: ::

	$ export HDF5_DIR=<dir>
	
so that pytables can find it.

Also install LZO and remember to do: ::

	$ export LZO_DIR=<dir>

cgtypes
-------------------------------------

Download cgkit-1.2.0 from sourceforge.

Use pyrex to recompile cgtypes: ::

	$ cd cgkit-1.2.0
	$ pyrex cgtypes.pyx
	$ python setup.py install


Finally
-------------------------------------

At this point: ::

	$ cd flydra
	$ python setup.py develop

Mac OS X
========

``motmot.FastImage`` cannot be installed, so you have to do a "light" installation.
