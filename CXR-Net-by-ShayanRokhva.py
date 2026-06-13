import torch
import torch.nn as nn


# ==============================
# 1) CBR Block
# ==============================

class CBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBRBlock, self).__init__()

        # Basic Conv -> BN -> ReLU block
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ==============================
# 2) Feature Extraction
# ==============================

class FeatureExtraction(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(FeatureExtraction, self).__init__()

        # Stage 1: 3 CBR blocks
        self.stage1_cbr1 = CBRBlock(in_channels, base_channels)
        self.stage1_cbr2 = CBRBlock(base_channels, base_channels)
        self.stage1_cbr3 = CBRBlock(base_channels, base_channels)

        # Stage 2: input = original input + stage 1 features
        self.stage2_cbr1 = CBRBlock(in_channels + base_channels, base_channels)
        self.stage2_cbr2 = CBRBlock(base_channels, base_channels)

        # Stage 3: input = previous concat output
        self.stage3_cbr1 = CBRBlock(in_channels + 2 * base_channels, base_channels)
        self.stage3_cbr2 = CBRBlock(base_channels, base_channels)

        # Stage 4
        self.stage4_cbr1 = CBRBlock(in_channels + 3 * base_channels, base_channels)
        self.stage4_cbr2 = CBRBlock(base_channels, base_channels)

        # Stage 5
        self.stage5_cbr1 = CBRBlock(in_channels + 4 * base_channels, base_channels)
        self.stage5_cbr2 = CBRBlock(base_channels, base_channels)

    def forward(self, x):
        # Stage 1
        s1_input = x
        x1 = self.stage1_cbr1(s1_input)
        x1 = self.stage1_cbr2(x1)
        x1 = self.stage1_cbr3(x1)
        out1 = torch.cat([s1_input, x1], dim=1)

        # Stage 2
        s2_input = out1
        x2 = self.stage2_cbr1(s2_input)
        x2 = self.stage2_cbr2(x2)
        out2 = torch.cat([s2_input, x2], dim=1)

        # Stage 3
        s3_input = out2
        x3 = self.stage3_cbr1(s3_input)
        x3 = self.stage3_cbr2(x3)
        out3 = torch.cat([s3_input, x3], dim=1)

        # Stage 4
        s4_input = out3
        x4 = self.stage4_cbr1(s4_input)
        x4 = self.stage4_cbr2(x4)
        out4 = torch.cat([s4_input, x4], dim=1)

        # Stage 5
        s5_input = out4
        x5 = self.stage5_cbr1(s5_input)
        x5 = self.stage5_cbr2(x5)
        out5 = torch.cat([s5_input, x5], dim=1)

        return out5


# ==============================
# 3) Classifier
# ==============================

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.5):
        super(Classifier, self).__init__()

        # Reduce spatial dimensions from (H, W) to (1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten (B, C, 1, 1) -> (B, C)
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_channels, hidden_dim1)
        self.relu1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(dropout_rate)

        # Final output layer
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        # Input x comes directly from FeatureExtraction
        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.dropout(x)
        x = self.fc3(x)

        return x
