perturbed_name = ['Linf', 'gauss', 'contrast', 'fog']
net_name_cifar = ['vgg', 'dense', 'res', 'mobile']
net_name_imagenet = ['shuffle', 'google', 'reg', 'mnas']

i = 0
data = [[[0. for ii in range(100)] for jj in range(100)] for kk in range(4)]
for j in range(4):
    source = 'image-multi\\' + perturbed_name[i] + '-' + net_name_imagenet[j] + '-multi.txt'
    print(source)
    f = open(source)
    lines = f.readlines()
    assert len(lines) == 100
    for k1 in range(100):
        tmp = lines[k1].split(sep='\t')
        for k2 in range(100):
            data[j][k1][k2] = float(tmp[k2])

standard = 0.0495  # threshold
result = [0] * 100
for k1 in range(100):  # 100 images
    for k2 in range(4):
        for k3 in range(k2 + 1, 4):
            former_big = False
            latter_big = False
            for k4 in range(100):  # 100 perturbation sizes
                if data[k2][k1][k4] - data[k3][k1][k4] > standard:
                    former_big = True
                elif data[k3][k1][k4] - data[k2][k1][k4] > standard:
                    latter_big = True
            if former_big and latter_big:
                result[k1] += 1
                print(k1, k2, k3)
print(result)
print([result.count(i) for i in range(7)])

