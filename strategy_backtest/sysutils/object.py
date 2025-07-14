import importlib
from typing import List

LIST_OF_RESERVED_METHOD_NAMES = ["log", "name", "parent", "description", "data"]

def resolve_rule(function):
    if hasattr(function, "__call__"):
        # it's a function, just return it so can be used
        # doesn't work for all functions, but we only care about callable ones
        return function

    if not isinstance(function, str):
        raise Exception(
            "Called resolve_function with non string or callable object %s"
            % str(function)
        )

    if "." in function:
        # it's another module, have to get it
        mod_name, func_name = function.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name, None)

    else:
        raise Exception(
            "Need full module file name string: %s isn't good enough"
            % function
        )

    return function



def hasallattr(some_object, attrlist: List[str]):
    """
    Check something has all the attributes you need
    :returns: bool
    """
    return all([hasattr(some_object, attrname) for attrname in attrlist])