import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch import Tensor
import numpy as np
import torch
from torch_geometric.data import Data

class OneHotEncoding0d(nn.Module):
    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)  # 7
        return torch.cat(
            [
                nn.functional.one_hot(x[..., i], cardinality)
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )

class TCM_Model(torch.nn.Module):
    def __init__(self, dropout, emb_dim, num_herb, cat_cardinalities, device):
        super(TCM_Model, self).__init__()
        self.device = device
        self.class_head1 = nn.Linear(805, 512)
        self.regression_head1 = nn.Linear(805, 512)
        self.class_head2 = nn.Linear(512, num_herb)
        self.regression_head2 = nn.Linear(512, num_herb)
        self.cat_module = OneHotEncoding0d(cat_cardinalities)

        # 1D-CNN 主干（对应 Keras Conv1D/AveragePooling1D/Flatten/Dense）
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=2, stride=1, padding=0)  # valid
        self.pool = nn.AvgPool1d(kernel_size=2)  # 若需 Max：改为 nn.MaxPool1d(2)

        # 根据 OH 长度计算展平维度，再映射到 805 维 pre
        self.input_length = 1350  # OH = 1+2+654+207+312+174
        conv_out_len = self.input_length - self.conv1.kernel_size[0] + 1  # 1349
        pooled_len = conv_out_len // 2  # 674
        backbone_dim = 10 * pooled_len   # 10 * 674 = 6740

        self.fc1 = nn.Linear(backbone_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 805)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X, bank, gat):
        x = []
        x.append(X[0])
        x.append(self.cat_module(X[1][:,:4]).float())
        symptom_OH = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        one_hot = torch.zeros(X[1].shape[0], 174, dtype=torch.float32, device=self.device)
        one_hot.scatter_(1, X[1][:, 4:].long(), 1)                #             1   2   3
        OH = torch.concat((symptom_OH, one_hot), dim=1) # 64, 1350 = 1+2+654+207+312+174

        # 1D-CNN 主干：Conv1D -> Pool -> Flatten -> Dense -> Dense -> pre(805)
        z = OH.unsqueeze(1)          # [batch, 1, 1350]
        z = self.conv1(z)            # Conv1D(filters=10, kernel_size=2, strides=1, padding='valid')
        z = self.pool(z)             # AveragePooling1D()
        z = torch.flatten(z, 1)
        z = F.relu(self.fc1(z)); z = self.dropout(z)
        pre = F.relu(self.fc2(z)); pre = self.dropout(pre)

        # 使用现有 head（不改 head 定义）
        class_hidden = self.class_head1(pre)        # 512
        regression_hidden = self.regression_head1(pre)   # 512
        class_output = self.class_head2(class_hidden)
        regression_output = self.regression_head2(regression_hidden)

        mask = torch.sigmoid(class_output)
        binary_mask = (mask > 0.8).float().detach()
        outputs = {
            'pred_logits': class_output,
            'pred_values': F.relu(regression_output * binary_mask)
        }
        return outputs
        # model = models.Sequential()
        # model.add(Conv1D(filters=10, kernel_size=2, padding='valid', kernel_initializer='uniform', strides=1))
        # if fusion == 'Avg':
        #     model.add(AveragePooling1D())
        # elif fusion == 'Max':
        #     model.add(MaxPooling1D())
        # else:
        #     pass
        # model.add(Flatten())
        # model.add(Dense(layer1, activation="relu"))
        # model.add(Dense(layer2, activation="relu"))
        # model.add(Dense(herblong, activation="softmax"))
        # model.build(input_shape=(2, max_sym_num, symlong))
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(),
        #     loss='binary_crossentropy',
        #     metrics=['accuracy']
        # )
        # model.summary()
