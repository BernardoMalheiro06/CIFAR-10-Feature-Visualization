import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1, bias = False)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.dropout = dropout

    def forward(self, x):
        x = self.conv1(x)

        x = self.batch_norm(x)

        x = torch.nn.functional.relu(x)
        
        x = torch.nn.functional.max_pool2d(x, kernel_size = 2, stride = 2)

        x = torch.nn.functional.dropout(x, p = self.dropout, training = self.training)

        return x



class ConvUpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output = False):
        super(ConvUpBlock, self).__init__()
        self.convTransp = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3 , stride = 1, padding = 1, bias = True)
        self.output = output

    def forward(self, x):
        x = self.convTransp(x)

        x = torch.nn.functional.relu(x)

        x = self.conv1(x)

        if not self.output:
            x = torch.nn.functional.relu(x)
        else:
            x = (torch.tanh(x) + 1) / 2

        return x



class Encoder(torch.nn.Module):
    def __init__(self, image_size, latent_dim):
        super(Encoder, self).__init__()
        self.conv_block1 = ConvBlock(3, 64, dropout=0.2)
        self.conv_block2 = ConvBlock(64, 128, dropout=0.2)
        self.dense1 = torch.nn.Linear(int(128 * (image_size / 4) ** 2), latent_dim * 2, bias = False)


    def forward(self, x):
        #Pass through convolutional blocks
        x = self.conv_block1(x) # (batch_size, 64, H/2, W/2)
        x = self.conv_block2(x) # (batch_size, 128, H/4, W/4)

        #Flatten
        x = x.view(x.size(0), -1) # (batch_size, 128 * (H/4) * (W/4))

        #First Dense Layer
        x = self.dense1(x) # (batch_size, latent_dim * 2)

        return x



class Decoder(torch.nn.Module):
    def __init__(self, image_size, latent_dim):
        super(Decoder, self).__init__()
        self.dense1 = torch.nn.Linear(latent_dim, int(128 * (image_size / 4) ** 2), bias = True)
        self.conv_block1 = ConvUpBlock(128, 64)
        self.conv_block2 = ConvUpBlock(64, 3, output=True)
        self.image_size = image_size


    def forward(self, x):
        #First Dense Layer
        x = self.dense1(x) # (batch_size, 128 * (H/4) * (W/4))

        x = torch.nn.functional.relu(x)

        #Reshape
        x = x.view(x.size(0), 128, int(self.image_size / 4), int(self.image_size / 4)) # (batch_size, 128, H/4, W/4)

        #Pass through convolutional blocks
        x = self.conv_block1(x) # (batch_size, 64, H/2, W/2)
        x = self.conv_block2(x) # (batch_size, 3, H, W)

        return x



class Classifier(torch.nn.Module):
    def __init__(self, image_size):
        super(Classifier, self).__init__()
        self.conv_block1 = ConvBlock(3, 32, dropout=0.2)
        self.conv_block2 = ConvBlock(32, 64, dropout=0.2)
        self.conv_block3 = ConvBlock(64, 128, dropout=0.2)
        self.dense1 = torch.nn.Linear(int(128 * (image_size / 8) ** 2), 256, bias = False)
        self.batch_norm = torch.nn.BatchNorm1d(256)
        self.dense2 = torch.nn.Linear(256, 10)


    def forward(self, x):
        #Pass through convolutional blocks
        x = self.conv_block1(x) # (batch_size, 32, H/2, W/2)
        x = self.conv_block2(x) # (batch_size, 64, H/4, W/4)
        x = self.conv_block3(x) # (batch_size, 128, H/8, W/8)

        #Flatten
        x = x.view(x.size(0), -1) # (batch_size, 128 * (H/8) * (W/8))

        #First Dense Layer
        x = self.dense1(x) # (batch_size, 256)

        #BatchNorm
        x = self.batch_norm(x)

        #ReLu Activation
        x = torch.nn.functional.relu(x)

        #Dropout
        x = torch.nn.functional.dropout(x, p = 0.5, training = self.training)

        #Second Dense Layer
        x = self.dense2(x)

        return x