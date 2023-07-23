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




class unet_conv_cfg(nn.Module):
    """
    Unet impleimentation for classifier free guidance
    """
    def __init__(self, c_in=3, c_out=3,time_dim=256, num_classes=None, device="cuda"):
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
        
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, inputs, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        if y is not None:
            t += self.label_emb(y)

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
    



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(3, 3, 64, 64).to(device)
    t = x.new_tensor([500] * x.shape[0]).long().to(device)
    model = unet_conv_cfg().to(device)
    y = model(x, t)
    print(y.shape)