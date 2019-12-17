import torch.nn as nn

class k1c2(nn.Module):
    def __init__(self, num_classes:int):
        """
        Convnet for music classification: 4 conv layers then 2 FC layers
        """
        
        super(k1c2, self).__init__()
                        
        net = nn.Sequential()
        
        def add_CNN_layer(num, size_in, size_out, kernel_size=3, padding=1, bn=True, pool_size=2):
            net.add_module( f"conv{num}", nn.Conv2d(size_in, size_out, kernel_size=3, stride=1, padding=padding) )
            if bn: 
                net.add_module( f"bn{num}",nn.BatchNorm2d(size_out) )
            if pool_size:
                net.add_module( f"pool{num}", nn.MaxPool2d(kernel_size=pool_size) )
        
        def add_FC_layer(num, size_in, size_out, dropout=0.5):
            net.add_module( f"fc{num}", nn.Linear(size_in, size_out))
            if dropout:
                net.add_module( f"dropout{num}", nn.Dropout(dropout))
        
        # for 0.5e6 parameter network (Table 1)
        add_CNN_layer(0, 1, 33) # 1 here is no of image channels
        add_CNN_layer(1, 33, 33)
        add_CNN_layer(2, 33, 66)
        add_CNN_layer(3, 66, 66)
        add_FC_layer(4, 66, 66)
        add_FC_layer(5, 66, 66)
        add_FC_layer(6, 66, num_classes)
        
        self.net = net
        
    def forward(self, input):
        return self.net(input)
