import json
"""
This file contains extra functions which are not part of the main data-processing
methods of the program.

These functions primarily deal with user input, and configuration of the processing
parameters.
"""


def short_match(key, options):
    """
    finds the value in 'options' which 'key' matches against, if any.
    If no or multiple matches are found, returns None.

    e.g: if the options are ["yes", "no", "new"], then
        'y', 'ye', or 'yes' will match against 'yes',
        but an input of 'n' will return None. ('no' will
        match 'no', and 'ne' or 'new' will match 'new')
    -----

    INPUTS:
        key: the key to search for matches to
        options: an iterable of options to find matches in
    """
    lkey = len(key)
    match = None
    for opt in options:
        if opt[:lkey] == key:
            if match is None:
                match = opt
            else:
                match = None
    return match


def ask(question, default=None, **options):
    """
    based on:
        http://stackoverflow.com/questions/3041986/python-command-line-yes-no-input

    prints a prompt to ask the user to choose from a number of options, with
    an optional default value which will be used if the user provides no input.
    -----

    INPUTS:
        question: The text-string to ask the user
        default: (optional) the option to return if no input is provided
        **options: the set of key/value pairs to choose from
    """
    if len(options) < 2:
        raise ValueError("A question needs to have at least 2 answers.")
    # create the prompt string
    if default is None:
        prompt = "[{}]".format('/'.join(o.lower() for o in options))
    # check that the default option is actually an option
    elif default not in options:
        raise ValueError("Invalid default answer: '%s'" % default)
    else:
        # capitalize the default prompt
        prompt = "[{}]".format(
            '/'.join(o.upper() if o == default else o.lower()
                     for o in options)
        )
    # keep looping until we get valid input
    while True:
        # print the prompt line
        print question, prompt,
        # get the user's response
        choice = raw_input().lower()
        # if there's a default, and no input has been entered...
        if default is not None and not choice:
            # ...return the default value
            return options[default]
        # otherwise, find the option that matches the user's input
        match = short_match(choice, options)
        # if it matches an option...
        if match is not None:
            # ...return that option's value
            return options[match]
        # otherwise, inform that the choice wasn't valid...
        else:
            print "Invalid or ambiguous choice.\n"
        # ...and the 'while' loop will ask the question again


def ask_yes_no(question, default="yes"):
    """
    A specific case of the 'ask' function, for asking yes/no questions.
    -----

    INPUTS:
        question: The text-string to ask the user
        default: (optional) the option to return if no input is provided
    """
    return ask(question, default, yes=True, no=False)


def parse_config(fname):
    """
    given the location of a configuration file, reads and interprets
    the contents of that file, returning a dict of the contents
    -----

    INPUTS:
        fname: The path to the configuration file
    """
    def boolean(string):
        if string == 'True':
            return True
        elif string == 'False':
            return False
        else:
            raise ValueError("'%s' is not a valid boolean value" % string)

    formats = {
        "grace_dmdt_method": str,
        "reconciliation_method": str,
        "random_walk": boolean,
        "verbose": boolean
    }
    output = {}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            # check if line is empty
            if not line:
                continue

            # check if it's a comment line
            if line[0] == '#':
                continue
            k, val = line.split()
            if k not in formats:
                raise ValueError("No such configuration parameter: " + k)
            func = formats[k]
            output[k] = func(val)
    return output

def load_json(fname):
    out = {}
    with open(fname) as f:
        out.update(json.load(f))
    return out