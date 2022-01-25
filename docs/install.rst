Installation Guide
==============================

Creating a conda environment (optional)
---------------------------------------

If you are using conda, you may wish to create a new environment
for the imbie processor, e.g::

    $ conda create --name imbie

you can then activate this environment using the command::

    $ conda activate imbie

If you are using conda, you should ensure that your environment
is active before continuing with the other steps on this page

Installing dependencies
-----------------------

To install the IMBIE processor using pip, first install the
dependencies listed in `requirements.txt` using the command::

    $ pip install -r requirements.txt

Installing the processor
------------------------

then, the processor itself can be installed using the command::

    $ python setup.py install

You can verify that the installation has succeeded by running::

    $ imbie --help
