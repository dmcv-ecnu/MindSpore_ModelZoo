import random
import numpy as np 
from PIL import Image 
import warnings
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
import mindspore
from utils.data_utils import construct_balanced_subset


class IncrementalDataset:
    def __init__(
        self,
        train_dataset,
        test_dataset,
        val_dataset=None,
        validation_split=0.0,
        random_order=None,
        base_classes=10,  # 基类个数
        increment=10,  # 增量阶段个数
    ):
        self.base_task_size = base_classes
        self.increment = increment
        self.increments = []
        self.random_order = random_order
        self.validation_split = validation_split

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self._setup_data()

        # 下面这些都不知道干什么用
        self.data_memory = None
        self.targets_memory = None
        self.idx_task_memory = None
        self.data_cur = None
        self.targets_cur = None
        self.idx_task_cur = None
        self.data_inc = None
        self.targets_inc = None
        self.idx_task_inc = None

        self._current_task = 0


    @property
    def n_tasks(self):
        return len(self.increments)    


    def new_task(self):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")
        min_class, max_class, x_train, y_train, x_test, y_test = self._get_cur_step_data_for_raw_data()
        self.data_cur, self.targets_cur = x_train, y_train
        if self.data_memory is not None:
            if len(self.data_memory) != 0:
                x_train = np.concatenate((x_train, self.data_memory))
                y_train = np.concatenate((y_train, self.targets_memory))
        self.data_inc, self.targets_inc = x_train, y_train
        self.data_test_inc, self.targets_test_inc = x_test, y_test
        trainset = self.get_custom_dataset("train")
        valset =  self.get_custom_dataset("val")
        testset =  self.get_custom_dataset("test")
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(x_train),
            "n_test_data": len(y_train),
        }
        self._current_task += 1
        return task_info, trainset, valset, testset
    

    def _get_cur_step_data_for_raw_data(self):
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        x_train, y_train = self._select_range(self.data_train, self.targets_train, low_range=min_class, high_range=max_class)
        x_test, y_test = self._select_range(self.data_test, self.targets_test, low_range=0, high_range=max_class)
        return min_class, max_class, x_train, y_train, x_test, y_test


    def _setup_data(self):
        # origin data.
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []
        self._split_dataset_task(self.train_dataset, self.val_dataset, self.test_dataset)
        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)


    def _split_dataset_task(self, train_dataset, val_dataset, test_dataset):
        increment = self.increment
        x_train, y_train = [], []
        for batch in train_dataset.create_dict_iterator(output_numpy=True):
            for image in batch["image"]:
                x_train.append(image)
            for label in batch["fine_label"]:
                y_train.append(label)
        x_train = np.stack(x_train, axis=0)
        y_train = np.stack(y_train, axis=0)

        # 此处进行划分验证集合，疑似缺失算子，先放弃  Soap
        x_val, y_val = np.array([]), np.array([])

        x_test, y_test = [], []
        for batch in test_dataset.create_dict_iterator(output_numpy=True):
            for image in batch["image"]:
                x_test.append(image)
            for label in batch["fine_label"]:
                y_test.append(label)
        x_test = np.stack(x_test, axis=0)
        y_test = np.stack(y_test, axis=0)
        # 给类标签换成指定id，方便控制训练类别的顺序
        if self.random_order is not None:
            self.class_order = order = self.random_order
            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)
        else:
            self.class_order = np.unique(y_train)
        if ((len(order) - self.base_task_size) % increment) != 0:
            print("Warning: not dividible")
        if self.base_task_size == 0:
            print("Warning: base task == 0")
        self.increments = [self.base_task_size] + [increment for _ in range((len(order) - self.base_task_size) // increment)]
        self.data_train.append(x_train)
        self.targets_train.append(y_train)
        self.data_val.append(x_val)
        self.targets_val.append(y_val)
        self.data_test.append(x_test)
        self.targets_test.append(y_test)


    @staticmethod
    def _map_new_class_index(y, order):
        return np.array(list(map(lambda x: order.index(x), y)))
    

    def _select_range(self, x, y, low_range=0, high_range=0):
        idxes = sorted(np.where(np.logical_and(y >= low_range, y < high_range))[0])
        if isinstance(x, list):
            selected_x = [x[idx] for idx in idxes]
        else:
            selected_x = x[idxes]
        return selected_x, y[idxes]
    

    def get_custom_dataset(self, data_source="train", balanced=False, oversample=False):
        assert data_source in ["train", "train_cur", "val", "test", "memory"]
        if data_source == "train":
            x, y = self.data_inc, self.targets_inc
        elif data_source == "train_cur":
            x, y = self.data_cur, self.targets_cur
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test_inc, self.targets_test_inc
        elif data_source == "memory":
            x, y = self.data_memory, self.targets_memory
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        if balanced:
            x, y = construct_balanced_subset(x, y, oversample)

        return DummyDataset(x, y)
    

class DummyDataset(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y, = self.x[idx], self.y[idx]
        return x, y