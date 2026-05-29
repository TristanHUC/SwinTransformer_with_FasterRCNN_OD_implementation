import torch.nn as nn

class FastRCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=4096, roi_output_size=(7, 7)):
        super(FastRCNN, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim * roi_output_size[0] * roi_output_size[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)

        self.cls_head = nn.Linear(hidden_dim, num_classes + 1) # +1 for background class

        self.bbox_head = nn.Linear(hidden_dim, (num_classes + 1) * 4)

    def forward(self, x):

    # reshape for the two fully connected layers
        x = x.flatten(start_dim=1)

        # fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Classification head
        class_logits = self.cls_head(x)

        # bounding boxes regression values
        bbox_deltas = self.bbox_head(x)

        return class_logits, bbox_deltas