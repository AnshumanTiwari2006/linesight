def _detect_environment() -> str:
    """
    Detect whether code is running inside Jupyter, Colab, or a plain script.

    Returns
    -------
    str
        One of: 'jupyter', 'colab', 'script'

    How it works
    ------------
    IPython injects get_ipython() into the global namespace when running
    inside a notebook or interactive shell. In a plain .py script, this
    name does not exist, so we catch the NameError.

    ZMQInteractiveShell is Jupyter/JupyterLab.
    Shell is Google Colab's kernel class name.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return 'jupyter'
        elif shell == 'Shell':
            return 'colab'
        else:
            return 'script'
    except NameError:
        return 'script'
