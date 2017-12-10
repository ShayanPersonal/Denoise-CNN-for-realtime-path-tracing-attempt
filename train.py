from __future__ import print_function
from math import log10
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

from model import DenoiseCNN
from data import get_dataset

def train(epoch):
    model.train()
    epoch_loss = 0
    for iteration, (input, target) in enumerate(training_data_loader, 1):
        input, target = Variable(input).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        prediction = model(input)
        loss = criterion(prediction, target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

    if epoch < 100 or epoch % 100 == 0:
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    scheduler.step(epoch_loss / len(training_data_loader))
    return input, prediction

def validate():
    model.eval()
    avg_psnr = 0
    for (input, target) in testing_data_loader:
        input, target = Variable(input).cuda(), Variable(target).cuda()
        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def test(model, boost_tensor):
    x = time.time()
    # preprocess
    #boost_tensor[:, :, :3] = torch.clamp(boost_tensor[:, :, :3], 0, 1)
    boost_tensor[:, :, 9] /=  torch.max(boost_tensor[:, :, 9]) + 0.00001
    boost_tensor[:, :, 10] /=  torch.max(boost_tensor[:, :, 10]) + 0.00001
    boost_tensor[:, :, 11] /=  torch.max(boost_tensor[:, :, 11]) + 0.00001
    boost_tensor[:, :, 12] /=  torch.max(boost_tensor[:, :, 12]) + 0.00001
    boost_tensor[:, :, 13] /=  torch.max(boost_tensor[:, :, 13]) + 0.00001
    # convert from (512, 512, 14) to (1, 14, 512, 512)
    boost_tensor = torch.unsqueeze(boost_tensor, 0)
    boost_tensor = boost_tensor.permute(0, 3, 1, 2)
    boost_tensor = torch.autograd.Variable(boost_tensor)
    # denoise
    output = model(boost_tensor)
    # transform back to (512, 512, 3)
    output = output.permute(0, 2, 3, 1)
    output = output.data
    print(x - time.time())
    return output[0]

def checkpoint(epoch, base_dir):
    model_out_path = "{}/model_epoch{}.pth".format(base_dir, epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def load_pretrained():
    cnn = torch.load("/home/moejoe/cs240a/cuda-pathtrace/denoise_cnn/results/12640364/model_epoch9000.pth").cuda()
    cnn.eval()
    return cnn

if __name__ == "__main__":
    print('===> Loading datasets')
    train_set, test_set = get_dataset()
    training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=4, shuffle=True, drop_last=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1, shuffle=False)

    print('===> Building model')
    model = DenoiseCNN().cuda()
    criterion = nn.L1Loss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1000, verbose=True, threshold=1e-4)

    time_str = str(int(time.time()))[2::]
    base_dir = "results/" + time_str

    for epoch in range(1, 400001):
        input, prediction = train(epoch)
        if epoch % 1000 == 0:
            if not os.path.exists(base_dir):
                os.mkdir(base_dir)
            prediction = np.swapaxes(prediction.cpu().data.numpy()[0], 0, 2)
            input = np.swapaxes(input.cpu().data.numpy()[0], 0, 2)
            plt.imsave("{}/in{}.png".format(base_dir, epoch), np.clip(input[:, :, :3], 0, 1))
            plt.imsave("{}/out{}.png".format(base_dir, epoch), prediction, 0, 1)
            if epoch % 1000 == 0:
                checkpoint(epoch, base_dir)
        #validate()