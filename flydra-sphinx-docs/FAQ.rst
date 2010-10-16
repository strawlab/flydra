


Frequently Asked Questions
=============================


The ``timestamp`` field is all wrong!
-------------------------------------

The ``timestamp`` field for 3D reconstructions is not the time when
the data was taken, but when it was done processing.  To get
timestamps spaced at the inter-frame interval, use ``frame *
(1.0/fps)``.


I cannot read the ``.kh5`` files
--------------------------------

Make sure you have installed the pytables-specific LZO compression filter.
