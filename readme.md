# IMBIE Processor

## Installation

### Installing Dependancies

The IMBIE processor requires some additional python libraries. A
list of the these can be found in the file `requirements.txt`.

These libraries should be installed before installing the IMBIE
processor. This can be done via `pip`, the python package manager,
using the command:

    pip3 install -r requirements.txt

### Installing the Processor

To install the imbie processor, use the command:

    python3 setup.py install

You may need administrator permissions to run the installation.


## Running the Processor

Once installed, the processor can be run using the command

    imbie [CONFIG]

Where CONFIG is the path to a configuration file. Details of the
configuration format and parameters can be found in the Software
User Manual.
