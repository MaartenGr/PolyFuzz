class NotInstalled:
    """
    This object is used to notify the user that additional dependencies need to be
    installed in order to use the string matching model.
    """

    def __init__(self, tool, dep):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        msg += f"pip install polyfuzz[{self.dep}]\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)
