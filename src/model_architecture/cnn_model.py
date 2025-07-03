import torch.nn as nn
import torch

class CustomCnnModel(nn.Module):
  def __init__(self,input_dim, num_classes):
    super(CustomCnnModel, self).__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes

    self.conv_layer =nn.Sequential(
        # c1
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        # 128x128x3 ---> 3x3x3x32 ---> wxhx32
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # c2
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # c3
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # c4
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self._to_linear = None
    self._get_conv_output(input_dim)

    self.fc_layer = nn.Sequential(
        nn.Linear(self._to_linear, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

  def _get_conv_output(self, input_dim = 128):
    with torch.no_grad():
      dummy_input = torch.zeros(1, 3, input_dim,input_dim)
      output = self.conv_layer(dummy_input)
      self._to_linear =  output.view(1, -1).size(1)


  def forward(self,x):
    x = self.conv_layer(x)
    x = x.view(x.size(0),-1)
    x = self.fc_layer(x)
    return x