import os
import pickle
import mindspore as ms
from mindspore.dataset.vision import py_transforms as transforms
from mindspore.dataset.transforms.py_transforms import Compose
import mindspore.ops as ops
from mindspore import Tensor
from PIL import Image
import numpy as np
import random


class MiniImageNet:
    def __init__(self, root, partition='train'):
        super(MiniImageNet, self).__init__()
        self.__iter = 0
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = Compose([transforms.RandomCrop(84, padding=4),
                                      # lambda x: np.asarray(x),
                                      transforms.ToTensor(),
                                      normalize])
        else:  # 'val' or 'test' ,
            self.transform = Compose([# lambda x: np.asarray(x),
                                      transforms.ToTensor(),
                                      normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        dataset_path = os.path.join(self.root,
                                    'mini/mini_imagenet/compacted_dataset/mini_imagenet_{}.pickle' \
                                    .format(self.partition))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        # for each class
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = Image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                # image_data = np.array(image_data, dtype='float32')

                # image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            # cx = np.arange(5)
            # np.random.shuffle(cx)
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])[0]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = \
                        self.transform(class_data_list[num_shots + i_idx])[0]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = ops.Stack(1)([Tensor(data, ms.float32) for data in support_data])
        support_label = ops.Stack(1)([Tensor(label, ms.float32) for label in support_label])
        query_data = ops.Stack(1)([Tensor(data, ms.float32) for data in query_data])
        query_label = ops.Stack(1)([Tensor(label, ms.float32) for label in query_label])

        return [support_data, support_label, query_data, query_label]

