import torch
import random

def gpu_turbo(eps):
    a=[]
    dev_cnt = torch.cuda.device_count()
    for i in range(dev_cnt):
        # 250 acounts for 20% single GPU
        a.append(torch.rand(500, 2500).to('cuda:'+str(i)))

    while True:
        if random.random() < eps:
            for i in range(dev_cnt):
                b = torch.sin(a[i])

if __name__ == '__main__':
    gpu_turbo(0.1)
