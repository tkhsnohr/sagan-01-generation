import torch
import torch.nn as nn

from einops import rearrange

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class SelfAttention(nn.Module):
    """Self-Attention Layer"""

    def __init__(self, in_dim: int) -> None:
        super(SelfAttention, self).__init__()

        # Prepare pointwise convolution with 1x1 kernel for query, key, and value
        self.query_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1,
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim // 8,
            kernel_size=1,
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=1,
        )

        # Softmax normalization for creating Attention Map
        self.softmax = nn.Softmax(dim=-2)

        # Coefficient for adding the original input x and the Self-Attention Map o
        # output = x + gamma * o
        # Initially, gamma is set to 0 and will be learned during training
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input variable
        X = x

        # Apply convolution and reshape: B,C,W,H -> B,C,N
        # Shape: B,C,N
        proj_query = rearrange(self.query_conv(X), "b c h w -> b c (h w)")
        # Transpose operation
        proj_query = proj_query.permute(0, 2, 1)
        # Shape: B,C,N
        proj_key = rearrange(self.key_conv(X), "b c h w -> b c (h w)")

        # Matrix multiplication
        # bmm performs batch matrix multiplication
        # Shape: B, N, N
        S = torch.bmm(proj_query, proj_key)

        # Normalization
        # Softmax function to make the sum of each row equal to 1
        attention_map_T = self.softmax(S)
        # Transpose the attention map
        attention_map = attention_map_T.permute(0, 2, 1)

        # Calculate the Self-Attention Map o
        # Size: B,C,N
        proj_value = rearrange(self.value_conv(X), "b c h w -> b c (h w)")
        # Multiply with the transposed attention map
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        # Align the tensor size of the Self-Attention Map o with X and output it
        o = o.view(*X.shape)
        out = x + self.gamma * o

        return out, attention_map


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int = 20,
        image_size: int = 64,
        image_channels: int = 1,
    ) -> None:
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    z_dim,
                    image_size * 8,
                    kernel_size=4,
                    stride=1,
                )
            ),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    image_size * 8,
                    image_size * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    image_size * 4,
                    image_size * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(inplace=True),
        )

        # Add Self-Attention layer
        self.self_attention1 = SelfAttention(in_dim=image_size * 2)

        self.layer4 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    image_size * 2,
                    image_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
        )

        # Add Self-Attention layer
        self.self_attention2 = SelfAttention(in_dim=image_size)

        self.last = nn.Sequential(
            nn.ConvTranspose2d(
                image_size,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        image_channels: int = 1,
    ) -> None:
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.Conv2d(
                    image_channels,
                    image_size,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.Conv2d(
                    image_size,
                    image_size * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.Conv2d(
                    image_size * 2,
                    image_size * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Add Self-Attention layer
        self.self_attention1 = SelfAttention(in_dim=image_size * 4)

        self.layer4 = nn.Sequential(
            # Add Spectral Normalization
            nn.utils.spectral_norm(
                nn.Conv2d(
                    image_size * 4,
                    image_size * 8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Add Self-Attention layer
        self.self_attention2 = SelfAttention(in_dim=image_size * 8)

        self.last = nn.Conv2d(image_size * 8, 1, kernel_size=4, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2
