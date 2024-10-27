

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images(took ages to complete)
def load_image(image_path, transform=None, max_size=400, shape=None):
    image = Image.open(image_path)
    if max_size:
        size = max(image.size)
        if size > max_size:
            size = max_size
            image.thumbnail((size, size), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

content_image = load_image("path/to/your/content/image.jpg", transform)
style_image = load_image("path/to/your/style/image.jpg", transform, shape=[content_image.size(2), content_image.size(3)])

# Display images
def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.show()

imshow(content_image, title='Content Image')
imshow(style_image, title='Style Image')

# Load pre-trained VGG19 model and freeze its parameters(crashed 10 times)
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Define layers to use for content and style loss
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Extract features
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Compute Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Define content loss and style loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Create a model with content and style loss layers(Works about 70% of the time and crashes you computer the rest the rest)
def get_style_model_and_losses(vgg, style_img, content_img, content_layers, style_layers):
    content_losses = []
    style_losses = []

    model = nn.Sequential()
    gram = GramMatrix()

    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, content_losses, style_losses

# Input image to be optimized
input_img = content_image.clone()

# Get model and loss functions
model, content_losses, style_losses = get_style_model_and_losses(vgg, style_image, content_image, content_layers, style_layers)
input_img.requires_grad_(True)
model.requires_grad_(False)

# Optimizer
optimizer = optim.LBFGS([input_img])

# Style transfer process()
run = [0]
while run[0] <= 300:
    def closure():
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print("run {}:".format(run))
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
            print()

        return style_score + content_score

    optimizer.step(closure)

# Save and display the final output
save_image(input_img, 'output.jpg')
imshow(input_img, title='Output Image')
