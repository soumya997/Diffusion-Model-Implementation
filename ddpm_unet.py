import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_c
            ),
        )

    def forward(self, inputs, t):
        x = self.conv(inputs)
        p = self.pool(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, p.shape[-2], p.shape[-1])
        p = p + emb
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=256):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_c
            ),
        )

    def forward(self, inputs, skip, t):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)



class unet_conv(nn.Module):
    def __init__(self, c_in=3, c_out=3,time_dim=256, device="cuda"):
        super().__init__()

        self.device = device
        self.time_dim = time_dim


        self.e1 = encoder_block(c_in, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)


        self.b = conv_block(512, 1024)


        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)


        self.outputs = nn.Conv2d(64, c_out, kernel_size=1, padding=0)

    def forward(self, inputs, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)


        s1, p1 = self.e1(inputs, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        s4, p4 = self.e4(p3, t)


        b = self.b(p4)


        d1 = self.d1(b, s4, t)
        d2 = self.d2(d1, s3, t)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)


        outputs = self.outputs(d4)

        return outputs
    

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    

class UNet_MHA(nn.Module):
    def __init__(self, c_in=3, c_out=3,time_dim=256, device="cuda"):
        super().__init__()

        self.device = device
        self.time_dim = time_dim

        """ Encoder """
        self.e1 = encoder_block(c_in, 64)
        self.sa1 = SelfAttention(64, 64)    

        self.e2 = encoder_block(64, 128)
        self.sa2 = SelfAttention(128, 32)
        
        self.e3 = encoder_block(128, 256)
        self.sa3 = SelfAttention(256, 16)
        
        self.e4 = encoder_block(256, 512)
        self.sa4 = SelfAttention(512, 8)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.sa5 = SelfAttention(512, 8)

        self.d2 = decoder_block(512, 256)
        self.sa6 = SelfAttention(256, 16)

        self.d3 = decoder_block(256, 128)
        self.sa7 = SelfAttention(128, 32)

        self.d4 = decoder_block(128, 64)
        self.sa8 = SelfAttention(64, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, c_out, kernel_size=1, padding=0)

    def forward(self, inputs, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        """ Encoder """
        s1, p1 = self.e1(inputs, t)
        # p1 = self.sa1(s1)

        s2, p2 = self.e2(p1, t)
        p2 = self.sa2(p2)
        
        s3, p3 = self.e3(p2, t)
        p3 = self.sa3(p3)

        s4, p4 = self.e4(p3, t)
        p4 = self.sa4(p4)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4, t)
        # print("size: ", d1.shape)
        # d1 = self.sa5(d1)


        d2 = self.d2(d1, s3, t)
        d2 = self.sa6(d2)

        d3 = self.d3(d2, s2, t)
        d3 = self.sa7(d3)
        
        d4 = self.d4(d3, s1, t)
        d4 = self.sa8(d4)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs
    

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

if __name__ == "__main__":
    # inputs = torch.randn((2, 32, 256, 256))
    # e = encoder_block(32, 64)
    # x, p = e(inputs)
    # print(x.shape, p.shape)
    #
    # d = decoder_block(64, 32)
    # y = d(p, x)
    # print(y.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # inputs = torch.randn((2, 3, 512, 512))
    x = torch.randn(3, 3, 64, 64).to(device)
    t = x.new_tensor([500] * x.shape[0]).long().to(device)
    model = unet_conv().to(device)
    y = model(x, t)
    print(y.shape)