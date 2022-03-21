class Recorder:
    def record(self, store):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def get_records(self):
        raise NotImplementedError
