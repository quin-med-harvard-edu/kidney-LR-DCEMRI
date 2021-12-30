import torch.nn as nn


class GenConvNet(nn.Module):

    def __init__(self):
        super(GenConvNet, self).__init__()

        model_ = []
        model_.append(nn.Conv2d(in_channels=1, out_channels=128,
                                kernel_size=3, padding=(1, 1), stride=(1, 1)))
        model_.append(nn.BatchNorm2d(num_features=128))
        model_.append(nn.ReLU())
        model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=(1, 1), stride=(1, 1)))
        model_.append(nn.BatchNorm2d(num_features=128))
        model_.append(nn.ReLU())
        model_.append(nn.UpsamplingNearest2d(scale_factor=2))

        for idx_block in range(4):
            model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                    kernel_size=3, padding=(1, 1), stride=(1, 1)))
            model_.append(nn.BatchNorm2d(num_features=128))
            model_.append(nn.ReLU())
            model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                    kernel_size=3, padding=(1, 1), stride=(1, 1)))
            model_.append(nn.BatchNorm2d(num_features=128))
            model_.append(nn.ReLU())
            model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                    kernel_size=3, padding=(1, 1), stride=(1, 1)))
            model_.append(nn.BatchNorm2d(num_features=128))
            model_.append(nn.ReLU())
            model_.append(nn.UpsamplingNearest2d(scale_factor=2))

        model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=(1, 1), stride=(1, 1)))
        model_.append(nn.BatchNorm2d(num_features=128))
        model_.append(nn.ReLU())
        model_.append(nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=3, padding=(1, 1), stride=(1, 1)))
        model_.append(nn.BatchNorm2d(num_features=128))
        model_.append(nn.ReLU())
        model_.append(nn.Conv2d(in_channels=128, out_channels=2,
                                kernel_size=3, padding=(1, 1), stride=(1, 1)))
        self.model = nn.ModuleList(model_)

    def forward(self, x):
        x_ = x
        for module in self.model:
            x_ = module.forward(x_)

        return x_


class DenseMapNet(nn.Module):

    def __init__(self, dim_input):
        super(DenseMapNet, self).__init__()

        model_ = []
        model_.append(nn.Linear(dim_input, 512))
        model_.append(nn.ReLU())
        model_.append(nn.Linear(512, 49))
        self.model = nn.ModuleList(model_)

    def forward(self, x):
        x_ = x
        for module in self.model:
            x_ = module.forward(x_)

        return x_
