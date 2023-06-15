Executing the processor
=======================
Once the IMBIE processor has been installed as per the instructions in Section 3, it can be invoked using the command ‘imbie’. This will execute the processor in the current directory.
The general interface to the processor is in the form: ::

    $ imbie [configuration file]

The argument provided should be the path to a valid IMBIE configuration file. The format of this file should follow the description in Section 4.2.
If the execution is successful, the output files will be found in the output directory that has been indicated in the configuration file.
Two additional processing utilities are also installed by the setup.py installation script. These are the dM-only processor, and the data pre-processor, which can be invoked by the commands imbie-processdm and imbie-preproc respectively. Each of these commands require an IMBIE configuration file as their only argument.
The dM-only processor will perform the IMBIE processing chain on change-of-mass files which have been created by the pre-processing tool.