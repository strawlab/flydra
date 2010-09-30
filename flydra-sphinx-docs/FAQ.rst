


Frequently Asked Questions
=============================


The ``timestamp`` field is all wrong!
-------------------------------------

The ``timestamp`` field is not the time when the data was taken, but when it was processed. 
To get the data timestamp, use ``frame * (1/60.0)``.


I cannot read the ``.kh5`` files
--------------------------------

Make sure you have installed the pytables-specific LZO compression filter.