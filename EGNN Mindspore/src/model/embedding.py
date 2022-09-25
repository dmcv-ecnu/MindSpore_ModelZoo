import mindspore.nn as nn


# encoder for imagenet dataset
class EmbeddingImagenet(nn.Cell):
    def __init__(self, emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.SequentialCell(nn.Conv2d(in_channels=3,
                                                  out_channels=self.hidden,
                                                  kernel_size=3,
                                                  padding=1,
                                                  pad_mode='pad',
                                                  has_bias=False),
                                        nn.BatchNorm2d(num_features=self.hidden, use_batch_statistics=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.LeakyReLU(alpha=0.2))

        self.conv_2 = nn.SequentialCell(nn.Conv2d(in_channels=self.hidden,
                                                  out_channels=int(self.hidden*1.5),
                                                  kernel_size=3,
                                                  padding=1,
                                                  pad_mode='pad',
                                                  has_bias=False),
                                        nn.BatchNorm2d(num_features=int(self.hidden*1.5), use_batch_statistics=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.LeakyReLU(alpha=0.2))

        self.conv_3 = nn.SequentialCell(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                                  out_channels=self.hidden*2,
                                                  kernel_size=3,
                                                  padding=1,
                                                  pad_mode='pad',
                                                  has_bias=False),
                                        nn.BatchNorm2d(num_features=self.hidden * 2, use_batch_statistics=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.LeakyReLU(alpha=0.2),
                                        nn.Dropout(0.6))

        self.conv_4 = nn.SequentialCell(nn.Conv2d(in_channels=self.hidden*2,
                                                  out_channels=self.hidden*4,
                                                  kernel_size=3,
                                                  padding=1,
                                                  pad_mode='pad',
                                                  has_bias=False),
                                        nn.BatchNorm2d(num_features=self.hidden * 4, use_batch_statistics=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.LeakyReLU(alpha=0.2),
                                        nn.Dropout(0.5))

        self.layer_last = nn.SequentialCell(nn.Dense(in_channels=self.last_hidden * 4,
                                            out_channels=self.emb_size, has_bias=True),
                                            nn.BatchNorm1d(self.emb_size, use_batch_statistics=True))

    def construct(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.shape[0], -1))
