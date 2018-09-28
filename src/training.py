import os
from PIL import Image
import numpy as np
import sys
import argparse
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc

import torchvision
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import utils
from models import Bottleneck, coordRegressor, HourglassNet


def lossHourglass(output, tensors):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_layer1 = loss_fn(output[0], tensors[0])  
    loss_layer2 = loss_fn(output[1], tensors[1])           
    loss_layer3 = loss_fn(output[2], tensors[2])   
    loss_layer4 = loss_fn(output[3], tensors[3])    

    # Adding the losses from the four layers in the Hourglass Network
    total_loss = loss_layer1 + loss_layer2 + loss_layer3 + loss_layer4 

    return total_loss

def lossCoordinate(pred, input_pts):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    return loss_fn(pred, input_pts)

def logTensorBoard(epoch, loss, model):
    logger.scalar_summary("loss", loss, epoch)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)


    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
  
class JVCRDataSet(Dataset):

    def __init__(self, path):
    
        self.path = path
        items = os.listdir(path)
        items.sort()

        datasetMap = {a:b for a,b in enumerate([i[:-4] for i in items if i.endswith('.pts')])} # maps index with imageid
        self.datasetMap = datasetMap
      

    def __getitem__(self, index):
      return self.datasetMap[index]
     

    def __len__(self):
        return len(self.datasetMap)

def main(args):
    if args.mode == 'pre-train':
        train(args)
    elif args.mode == 'train':
        print('code to be added for full training..') # TBD
    else:    
        download_dataset(args)    


def download_dataset(args):
    #file_id = '1zZ6B3r8H2XrvT96GAfCpgzkSGZMSLtIU' # test doc
    file_id = '1kQihg2Yfc2clM5Qavxh2RiGc2EIg-4bX' # afwlp-2000 - 2GB
    destination = args.dataset_path + args.dataset_file_name + ".tar"
    utils.dataset_utils.download_file_from_google_drive(file_id, destination)
    utils.dataset_utils.extract_tar_file(destination, args.dataset_path)


def train(args):
    num_stacks=1
    num_blocks=4
    num_classes=[1, 2, 4, 64]
    
    logger = Logger(args.logdir)
    learning_rate = args.learning_rate
    save_freq = args.model_save_freq
    num_epochs = args.epochs
    batch_size = args.data_loader_batch_size
    dataset_path = args.dataset_path + args.dataset_file_name
    pre_train_mode = args.pre_train_mode
    optimizer = args.optimizer

    if pre_train_mode == 'hourglass': 
        print('hourglass-training..')
        model = HourglassNet(Bottleneck, num_blocks, num_stacks, num_classes)
    else:
        print('coordinate-regression-training..')
        model = coordRegressor(68)

    train_dataset = JVCRDataSet(dataset_path)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    #optimizer
    if optimizer == "ADAM":
        print('ADAM optimizer applied')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif optimizer == "SGD":    
        if args.momentum != 0 :
            print('SGD with momentum optimizer applied')
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum)
        else:
            print('SGD without momentum optimizer applied')
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    else:
        print('RMSPROP optimizer applied')
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)


    counter = 0
    for epoch in range(num_epochs):

        for batch_idx, (x) in enumerate(train_loader):
            counter = counter + 1
            images=[]
            hg_tensors=[] # gt tensor for hourglass
            reg_tensors=[] # gt tensor for coord regression
            pts=[]

            for item in x:
              images.append(Image.open(dataset_path+item+".jpg").convert('RGB'))
              hg_tensors.append(torch.load(dataset_path+item+".tensor"))
              reg_tensors.append(torch.load(dataset_path+item+".tensor")[-1].squeeze())
              pts.append(utils.transform_utils.to_torch(utils.transform_utils.readPtsTorch(dataset_path+item+".pts")))        
            images = utils.transform_utils.inputListTransform(images, pts)
            
            #compute losses
            if pre_train_mode == "hourglass":
                tensors = utils.transform_utils.reshapeTensorList(hg_tensors)
                output = model(images)
                loss = lossHourglass(output, tensors)
            else: # coordinate regression
                input_tensor = torch.stack(reg_tensors)
                input_tensor = input_tensor.unsqueeze(1)
                input_pts = torch.stack(pts).view(batch_size,-1)
                y_pred, y_pred1, y_pred_2 = model(input_tensor)
                loss = lossCoordinate(y_pred_2, input_pts)

            print("epoch",epoch, "batch_number", batch_idx, "loss", loss.item())
          
            model.zero_grad()

            loss.backward()
            optimizer.step()
                  
        if epoch % save_freq == 0:
            logTensorBoard(epoch, loss, model)
            file_name = 'epoch-'+str(epoch)+'-loss-'+str(loss.item())+'-'+pre_train_mode+'-model-opt.model'
            utils.transform_utils.save_model(model, optimizer, file_name)
            print('model '+file_name+' saved..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Voxel and Coordinate Regression')

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate for the model')
    parser.add_argument('-o', '--optimizer', default='eg. RMSPROP, ADAM', type=str, help='optimizer to use. eg. ADAM, RMSPROP, SGD')
    parser.add_argument('-m', '--momentum', default=0, type=float, help='data set name') 
    parser.add_argument('-s', '--model_save_freq', default=5, type=int, help='checkpoint save frequency(# of epochs)')
    parser.add_argument('-e', '--epochs', default=25, type=int, help='# of epochs')
    parser.add_argument('-dl', '--data_loader_batch_size', default=4, type=int, help='data loader batch size')
    parser.add_argument('-mode', '--mode', default='download', type=str, help='modes -> download_dataset, pre-train, train') 
    parser.add_argument('-pre_train_mode', '--pre_train_mode', default='hourglass', type=str, help=' eg. hourglass, coordinate. this is for individual training of the models') 
    # should default be /data
    parser.add_argument('-dspath', '--dataset_path', default='/tmp/', type=str, help='data set path') # TBD change to /tmp
    parser.add_argument('-dsname', '--dataset_file_name', default='JVCR-AFLW200-Dataset', type=str, help='data set name') 
    parser.add_argument('-logdir', '--logdir', default='/tmp', type=str, help='log directory for tensorboard') 

    #push this constant out of code
    logger = Logger('/tmp/JVCR/logs')

    main(parser.parse_args())

# things to try.. try the change in path.. pretrain model change.. try train param- for logging.. try logtensorboard.. try dataset download.. different optimizer

# download python training.py
# pre-training python training.py --mode=pre-train --pre_train_mode=hourglass --optimizer=SGD --momentum=0.1
# python training.py --mode=pre-train --pre_train_mode=hourglass --optimizer=RMSPROP
