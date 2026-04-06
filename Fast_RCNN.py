import torch.nn as nn

class FastRCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=1024, roi_output_size=(7, 7)):
        super(FastRCNN, self).__init__()

        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(3,3), stride=1, padding='same')
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1,1), stride=1, padding='same')
        self.relu = nn.ReLU()

        self.cls_head = nn.Linear(hidden_dim*roi_output_size[0]*roi_output_size[1], num_classes + 1) # +1 for background class

        self.bbox_head = nn.Linear(hidden_dim*roi_output_size[0]*roi_output_size[1], (num_classes + 1) * 4)

    def forward(self, x):

        # convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # reshape for the two fully connected layers
        x = x.flatten(start_dim=1)

        # Classification head
        class_logits = self.cls_head(x)

        # bounding boxes regression values
        bbox_deltas = self.bbox_head(x)

        return class_logits, bbox_deltas