
class S3DISConfig:
    device = None

    num_points = 40960  # Number of input points
    num_classes = 13  # Number of valid classes

    batch_size = 1  # batch_size during training
    train_steps = 500  # Number of steps per epochs

    val_batch_size = 32  # batch_size during validation and test
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension
    num_layers = len(sub_sampling_ratio)

    noise_init = 3.5  # noise initial parameter
    max_epoch = 5#0  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate

