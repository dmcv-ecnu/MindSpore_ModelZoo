class MemorySize:
    def __init__(self, mode, total_memory=None, fixed_memory_per_cls=None):
        self.mode = mode
        assert mode.lower() in ["uniform_fixed_per_cls", "uniform_fixed_total_mem"]
        self.total_memory = total_memory
        self.fixed_memory_per_cls = fixed_memory_per_cls
        self._nb_classes = 0

    def update_nb_classes(self, nb_classes):
        self._nb_classes = nb_classes

    @property
    def mem_per_cls(self):
        if "fixed_per_cls" in self.mode:
            return [self.fixed_memory_per_cls] * self._nb_classes
        elif "fixed_total_mem" in self.mode:
            return [self.total_memory // self._nb_classes]* self._nb_classes
        
    @property
    def memsize(self):
        if self.mode == "fixed_total_mem":
            return self.total_memory
        elif self.mode == "fixed_per_cls":
            return self.fixed_memory_per_cls * self._nb_classes