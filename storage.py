class Storage:
    def __init__(self, data={}):
        super().__init__()
        if isinstance(data, dict):
            self.data = data
        else:
            raise Exception

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return None

    def remove(self, key):
        if key in self.data:
            self.data.pop(key)
        else:
            raise KeyError(f'Storage has no key {key}, nothing to remove')

    def set(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError(f'Storage has no key {key}, can\'t set value {value}')
    
    def add(self):
        pass
