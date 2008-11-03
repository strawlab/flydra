Getting started
===============

This document assumes you are using Ubuntu linux and that you have a
basic command line knowledge such as how to change directories, list
files, and so on.

We use virtualenv_ to create an installation environment for flydra
that does not overwrite any system files. Get the virtualenv source
code, upack it and create a "virtual Python installation" in your home
directory called "PY_flydra"::

  wget http://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.3.tar.gz
  tar xvzf virtualenv-1.3.tar.gz
  cd virtualenv-1.3
  python virtualenv.py ~/PY_flydra

.. _subversion: http://subversion.tigris.org/
.. _virtualenv: http://pypi.python.org/pypi/virtualenv

To download the development version of flydra, you need
subversion_. To install it, run::

  sudo apt-get install subversion

Now to download ("checkout") flydra into your current directory, type::

  svn checkout https://code.astraw.com/kookaburra/trunk/flydra

To build and install flydra to your virtual Python installation::

  sudo dpkg --purge python-flydra # remove system flydra to prevent confusion
  source ~/PY_flydra/bin/activate
  cd flydra
  python setup.py develop

You should now have a working flydra installation in your virtual
Python environment. To test this, type::

  python -c "import flydra.version; print flydra.version.__version__"

The output should be something like "0.4.28-svn", indicating this is
the development version after release 0.4.28.
