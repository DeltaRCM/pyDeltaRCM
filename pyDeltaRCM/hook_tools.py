import abc


class hook_tools(abc.ABC):
    """Tools defining various hooks in the model.

    For an explanation on model hooks, see the user guide.
    """

    def hook_import_files(self):
        """Hook called immediately before :obj:`import_files`.
        """
        pass
