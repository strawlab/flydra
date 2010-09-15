rm -rf ext
svn co -r 8333 http://svn.scipy.org/svn/numpy/trunk/doc/sphinxext ext
find ext -name '.svn' -print0 | xargs -0 rm -rf
