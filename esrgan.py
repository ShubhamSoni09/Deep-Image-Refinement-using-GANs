from PIL import Image
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19
from torchvision import transforms

device  = "cpu"

class ConvolutionalBlock(nn.Module):
    def __init__(self, inp, out, use_act, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Conv2d(
            inp,
            out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()
        
    def forward(self, x):
        return self.act(self.block(x))
    

class Upsample(nn.Module):
    def __init__(self, input, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(input, input, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))
    
class DenseResidual(nn.Module):
    def __init__(self, input, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        
        for i in range(5):
            self.blocks.append(
                ConvolutionalBlock(
                    input + channels*i,
                    channels if i <= 3 else input,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <=3 else False,
                )
            )
        
    def forward(self, x):
        x_new = x
        for block in self.blocks:
            out = block(x_new)
            x_new = torch.cat([x_new, out], dim=1)
        
        return self.residual_beta * out + x
    
class BlockRRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blockrrdb = nn.Sequential(*[DenseResidual(in_channels) for _ in range(3)])
    
    def forward(self, x):
        return self.blockrrdb(x) * self.residual_beta + x

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=1):  # Reduced from 23 to 1
        super().__init__()
        self.initial_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.residuals_layer = nn.Sequential(*[BlockRRDB(num_channels) for _ in range(num_blocks)])
        self.conv_layer = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples_layer = nn.Sequential(
            Upsample(num_channels),
            Upsample(num_channels)
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
    def forward(self, x):
        initial_layer = self.initial_layer(x)
        out = self.residuals_layer(initial_layer)
        out = self.conv_layer(out) + initial_layer
        out = self.upsamples_layer(out)
        out = self.final_layer(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(64, 64, 128, 128, 256, 256, 512, 512)):
        super().__init__()

        self.in_channels = in_channels
        self.features = list(features)

        blocks = []

        for idx, feature in enumerate(self.features):
            blocks.append(
                ConvolutionalBlock(
                    inp=in_channels,
                    out=feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_act=True
                )
            )
            in_channels=feature

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(in_features=512*6*6, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.blocks(x)
        out = self.classifier(features)
        return out
    
def init_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

class Loss(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        vgg_inputs = self.vgg(output)
        vgg_targets = self.vgg(target)
        loss = self.loss(vgg_inputs, vgg_targets)

        return loss
    
class GANLoss(nn.Module):
    def __init__(self, l1_weight=1e-2, vgg_weight=1, device='cpu'):
        super().__init__()
        
        self.vgg_loss = Loss(device)
        self.l1_loss = nn.L1Loss().to(device)
        self.l1_weights = l1_weight
        
    def forward(self, output, target):
        vgg_loss = self.vgg_loss(output, target)
        l1_loss = self.l1_loss(output, target)
        
        loss = vgg_loss + self.l1_weights * l1_loss
        return loss
    
def penalty_gradient(critic, actual, fake, device):
    batch_size, channels, height, width = actual.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated = actual * alpha + fake.detach() * (1 - alpha)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    gradients = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

num_channels = 3
generator = Generator(in_channels=num_channels, num_blocks=1).to(device)
discriminator = Discriminator(num_channels).to(device)
init_weights(generator)

generator_scaler = torch.amp.GradScaler('cuda')
discriminator_scaler = torch.amp.GradScaler('cuda')

generator.to(device)
generator.load_state_dict(torch.load("generator.pt", map_location=device))

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean  

def downsample_image(hr_image, scale_factor=4):
    hr_image = Image.open(hr_image).convert("RGB")
    hr_width, hr_height = hr_image.size
    lr_width, lr_height = hr_width // scale_factor, hr_height // scale_factor
    lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)
    return lr_image

def process_with_esrgan(filepath, gan2_path):
    try:
        # Clear memory before processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        low_res = downsample_image(filepath, 4)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor with range [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        low_res = transform(low_res).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_high_res = generator(low_res)
        
        fake_high_res = denormalize(fake_high_res)[0].permute(1, 2, 0).cpu().numpy()
        fake_high_res = Image.fromarray((fake_high_res * 255).astype('uint8'))
        fake_high_res.save(gan2_path)
        
        # Clear memory after processing
        del low_res, fake_high_res
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"ESRGAN processing error: {e}")
        # Save a copy of original image if processing fails
        original = Image.open(filepath)
        original.save(gan2_path)
