def rename_function(func, mod):
    # get current module
    # thismodule = sys.modules[__name__]
    # get current function name
    func_name = func.__name__
    # set new function name
    func_name = f"renamed_{func_name}"
    # set new name on current module
    setattr(mod, func_name, func)


REGISTERED_INTERFACES = dict()


def register_interface(func, name=""):
    """Register a transformation function (interface) between two representations of models.

    Parameters
    ----------
    func
        Function to be registered. Needs to start with "from_" or "to_" to signify the direction of transformation
    name
        (Optional) String that defines the name under which the function should be registered. If left empty,
        the name will be formatted in the form {target}_from_{source}, where target and source are representations to
        transform from or to."""
    if name is "":

        # get interface name from module name
        module_name = func.__module__.split(".")[-1]
        # parse to_ and from_ functions
        func_name = func.__name__
        if func_name.startswith("from_"):
            target = module_name
            source = func_name[5:]  # crop 'from_'
        elif func_name.startswith("to_"):
            source = module_name
            target = func_name[3:]  # crop 'to_'
        else:
            raise ValueError(f"Function name {func_name} does not adhere to convention to start "
                             f"with either `to_` or `from_`.")  # ignore any other functions
        new_name = f"{target}_from_{source}"
    else:
        new_name = name

    if new_name in REGISTERED_INTERFACES:
        raise ValueError(f"Interface {new_name} already exist. Cannot add {func}.")
    else:
        REGISTERED_INTERFACES[new_name] = func

    return func
