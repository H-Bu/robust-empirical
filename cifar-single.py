import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from vgg import VGG
from densenet import DenseNet121
from resnet import ResNet101
from mobilenetv2 import MobileNetV2


def DNN_sample(_net, x, per_gen, per_size, batch_size, total):
    x_raw = x
    count = 0
    for i in range((total - 1) // batch_size + 1):
        if i == (total - 1) // batch_size:
            num = total - batch_size * ((total - 1) // batch_size)
        else:
            num = batch_size
        per_x = per_gen(x_raw, per_size, num)
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        x = transform(x_raw)
        per_x = transform(per_x)
        _net.eval()
        with torch.no_grad():
            x, per_x = x.to(device), per_x.to(device)
            outputs_x = _net(x)
            _, predicted_x = outputs_x.max(1)
            outputs = _net(per_x)
            _, predicted = outputs.max(1)
            result = (predicted == predicted_x)
            count += result.count_nonzero().item()
    return count


def contrast(x, per_size, num):
    c = np.random.uniform(max(1-per_size-0.2, 0), min(1-per_size+0.2, 1), (num, 1, 1, 1))
    x = x.repeat(num, 1, 1, 1).cpu()
    means = np.mean(x.numpy(), axis=(2, 3), keepdims=True)
    return np.clip((x[0] - means[0]) * c + means, 0, 1).float()


def gauss_noise(x, per_size, num):
    x = x.repeat(num, 1, 1, 1)
    noise = (torch.randn(size=x.shape, device=x.device)*per_size)
    return torch.clamp(x + noise, 0, 1)


def L_inf(x, per_size, num):
    x = x.repeat(num, 1, 1, 1)
    return torch.clamp(x + (2*torch.rand(size=x.shape, device=x.device)-1) * per_size, 0, 1)


def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x, per_size, num):
    x = x.repeat(num, 1, 1, 1).cpu()
    max_val = x.max()
    for i in range(num):
        x[i] += per_size[0] * plasma_fractal(wibbledecay=per_size[1])[:32, :32][np.newaxis, ...]
    return np.clip(x * max_val / (max_val + per_size[0]), 0, 1).float()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# net = [VGG('VGG16'), DenseNet121(), ResNet101(), MobileNetV2()]
net = [MobileNetV2()]
net_num = len(net)
for i in range(net_num):
    net[i] = net[i].to(device)
    if device == 'cuda':
        net[i] = torch.nn.DataParallel(net[i])
        cudnn.benchmark = True
# weight = torch.load('net_weight/' + 'vgg16' + '_ckpt.pth')
# net[0].load_state_dict(weight['net'])
# net[0].eval()
# weight = torch.load('net_weight/' + 'densenet121' + '_ckpt.pth')
# net[0].load_state_dict(weight['net'])
# net[0].eval()
# weight = torch.load('net_weight/' + 'resnet101' + '_ckpt.pth')
# net[0].load_state_dict(weight['net'])
# net[0].eval()
weight = torch.load('net_weight/' + 'mobilenetv2' + '_ckpt.pth')
net[0].load_state_dict(weight['net'])
net[0].eval()


transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
dataiter = iter(testloader)

img_num = 100
perturbed_num = 100
sample_num = 1000
per_par_low = 0
per_par_high = 0.4
# per_par_low = (0.5,2)
# per_par_high = (2.5,1.5)
print('CIFAR-10', 'L_inf-single', 'mobile')
print('per_par_low =', per_par_low, 'per_par_high =', per_par_high)
print('img_num =', img_num, 'perturbed_num =', perturbed_num, 'sample_num =', sample_num)
step = (per_par_high - per_par_low) / perturbed_num
# step = ((per_par_high[0] - per_par_low[0])/100, (per_par_high[1] - per_par_low[1])/100)  # for fog
result_density = [[[0. for i in range(perturbed_num)] for j in range(img_num)] for k in range(net_num)]
for ii in range(img_num):
    while True:
        images, labels = dataiter.__next__()
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            images_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
            all_correct = True
            for i in range(net_num):
                if net[i](images_norm).max(1)[1][0] != labels:
                    all_correct = False
                    break
            if all_correct:
                break

    for j in range(net_num):
        for i in range(perturbed_num):
            per_par = per_par_low+step*(i+1)
            # per_par = (per_par_low[0] + step[0] * (i + 1), per_par_low[1] + step[1] * (i + 1))
            result_density[j][ii][i] = DNN_sample(net[j], images, L_inf, per_par, 100, sample_num) / sample_num

print(result_density)

