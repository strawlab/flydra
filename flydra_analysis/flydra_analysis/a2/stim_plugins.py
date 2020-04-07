import pkg_resources


class PluginLoader:
    def __init__(self):
        self.all_names = []
        pkg_env = pkg_resources.Environment()
        self.EP = "flydra.kdviewer.plugins"
        for name in pkg_env:
            egg = pkg_env[name][0]
            for name in egg.get_entry_map(self.EP):
                egg.activate()
                self.all_names.append(name)

    def __call__(self, draw_stim_func_str):
        pkg_env = pkg_resources.Environment()
        for name in pkg_env:
            egg = pkg_env[name][0]
            for name in egg.get_entry_map(self.EP):
                if name != draw_stim_func_str:
                    continue
                entry_point = egg.get_entry_info(self.EP, draw_stim_func_str)
                PluginFunc = entry_point.load()
                return PluginFunc
        raise ValueError(
            "Could not find stimulus drawing plugin. Available: %s"
            % (str(self.all_names),)
        )
