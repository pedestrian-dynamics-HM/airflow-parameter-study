def split_argument_list(argv, separator=" "):
    """
    Splits a list of command line arguments into its sub-parts. For example:
    argv    = ["--opt1 arg1 arg2", "--opt2 arg1"] is separated to
    argv_   = ["--opt1", "arg1", "arg2", "--opt2", "arg1"].

    The latter format is required when argv should be processed by argparse.
    """

    format_argv = False

    # Find any element in argv that contains <separator>
    for element in argv:
        if element.__contains__(separator):
            format_argv = True

    if format_argv:
        argv_ = separator.join(argv)
        argv_ = argv_.split(sep=separator)
    else:
        argv_ = argv
    return argv_


def _str_to_bool(arg):
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise ValueError("Boolean value expected.")
