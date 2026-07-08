Development notes
=================

Tutorials and examples are created using sphinx-gallery.

Sample data should be available at a URL or DOI somewhere. The plan is
ultimately to maintain a zenodo repository of sample data managed with
pooch. While developing new tutorials, local files can be picked up
and in-between DOI releases it is possible to stash data on Github as
outputs of a quasi-"release". Users should not be exposed to the details:
instead a wrapper function is provided in the main library.
