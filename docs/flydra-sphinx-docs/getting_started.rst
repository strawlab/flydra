Getting started with flydra development
=======================================

This document assumes you are using Ubuntu linux and that you have a
basic command line knowledge such as how to change directories, list
files, and so on. Flydra is written primarily in the Python_
language. The `Python tutorial`_ is highly recommended when getting
started with Python.

.. _Python: http://python.org
.. _Python tutorial: http://docs.python.org/tutorial/index.html

Using a Python virtual environment
----------------------------------

We use virtualenv_ to create an installation environment for flydra
that does not overwrite any system files. Get the virtualenv source
code, upack it and create a "virtual Python installation" in your home
directory called "PY_flydra"::

  wget https://pypi.python.org/packages/source/v/virtualenv/virtualenv-1.3.tar.gz
  tar xvzf virtualenv-1.3.tar.gz
  cd virtualenv-1.3
  python virtualenv.py ~/PY_flydra

.. _virtualenv: http://pypi.python.org/pypi/virtualenv

Getting and installing flydra from the source code repository
-------------------------------------------------------------

To download the development version of flydra, you need
git_. To install it, run::

  sudo apt-get install git-core gitk

.. _git: http://git-scm.org/

You will need to send Andrew Straw your ssh key following `these
instructions`__.

.. __: http://code.astraw.com/git-repos.html

Now, to download ("checkout") flydra into your current directory, type::

  git clone git@code.astraw.com:flydra.git

To build and install flydra to your virtual Python installation::

  sudo dpkg --purge python-flydra # remove system flydra to prevent confusion
  source ~/PY_flydra/bin/activate
  cd flydra
  python setup.py develop

You should now have a working flydra installation in your virtual
Python environment. To test this, type::

  cd ~ # switch to a directory without a flydra/ subdirectory
  python -c "import flydra_core.version; print flydra_core.version.__version__"

The output should be something like "0.4.28-svn", indicating this is
the development version after release 0.4.28.

Testing your installation
-------------------------

Finally, the full test suite may be run with nose_::

  source ~/PY_flydra/bin/activate
  easy_install nose
  nosetests flydra

.. _nose: http://somethingaboutorange.com/mrl/projects/nose/

.. _editing-the-docs:

Editing the documentation
-------------------------

This documentation is built with Sphinx_ 0.6.2.

.. _Sphinx: http://sphinx.pocoo.org/

To download and install sphinx into your virtual environment::

  easy_install sphinx

Now, to build the flydra documentation::

  cd /path/to/flydra/flydra-sphinx-docs/
  ./get-svn.sh
  make html

The documentation will be built in
/path/to/flydra/flydra-sphinx-docs/.build/html/index.html You may view
it with::

  firefox .build/html/index.html

Alternatively, you can automatically build and upload it by pushing
your changes to the central git repository and clicking the "Force
build" button on the "doc uploader" build slave.

Continuous integration
----------------------

We are running a Buildbot that builds and tests flydra. You can see
the waterfall display at http://flydra.bb.astraw.com/waterfall .
