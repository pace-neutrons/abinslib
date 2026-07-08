Examples
========

This is a collection of worked examples; they are executed when
documentation is built and converted into a gallery format.  They can
be read there or downloaded as executable notebooks for further
experimentation.

To run these examples, make sure abinslib was installed with the "tutorial" extra, e.g.::

  pip install abinslib[tutorial]

This will ensure the `pooch <https://pypi.org/project/pooch/>`_
library is available to download sample data. Data files are accessed
by name during tutorials using :func:`abinslib.data.get_data`; this
will download a copy the first time it is used and refer to a local
cache after that.
