from abc import ABCMeta, abstractmethod


class DataSeries(metaclass=ABCMeta):
    @property
    def min_time(self):
        return self._get_min_time()

    @property
    def max_time(self):
        return self._get_max_time()

    def __init__(self, user, user_group, data_group, basin_group, basin_id,
                 basin_area, computed=False, merged=False):
        self.user = user
        self.user_group = user_group
        self.data_group = data_group
        self.basin_group = basin_group
        self.basin_id = basin_id
        self.basin_area = basin_area

        self.computed = computed
        self.merged = merged

    def limit_times(self, min_t=None, max_t=None):
        if min_t is not None:
            self._set_min_time(min_t)
        if max_t is not None:
            self._set_max_time(max_t)

    @abstractmethod
    def _set_min_time(self, min_t):
        return

    @abstractmethod
    def _set_max_time(self, max_t):
        return

    @abstractmethod
    def _get_min_time(self):
        return

    @abstractmethod
    def _get_max_time(self):
        return
