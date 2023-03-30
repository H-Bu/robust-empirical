perturbed_name = ['Linf', 'gauss', 'contrast', 'fog']
net_name_cifar = ['vgg', 'dense', 'res', 'mobile']
net_name_imagenet = ['shuffle', 'google', 'reg', 'mnas']

i = 3
j = 3
data = [[0. for ii in range(100)] for jj in range(100)]
source = 'image-single\\'+perturbed_name[i] + '-' + net_name_imagenet[j] + '-single.txt'
print(source)
f = open(source)
lines = f.readlines()
assert len(lines) == 100
for k1 in range(100):
    tmp = lines[k1].split(sep='\t')
    for k2 in range(100):
        data[k1][k2] = float(tmp[k2])

for k1 in range(100):
    flag = False
    for k2 in range(100):
        for k3 in range(k2, 100):
            if data[k1][k3]-data[k1][k2] > 0.0495:
                print(k1)
                flag = True
                break
        if flag:
            break

