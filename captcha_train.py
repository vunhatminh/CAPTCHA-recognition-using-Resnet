import torch
from torch import optim
from torch import nn
from captcha_loader import Captcha
from torch.utils.data import DataLoader
import argparse
import string
import captcha_resnet

def arg_parse():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument(
            "--model", dest="model"
        )
    parser.add_argument(
            "--numchar", dest="numchar", type=int
        )
    parser.add_argument(
            "--lr", dest="lr", type=float
        )
    parser.add_argument(
            "--epoch", dest="epoch", type=int
        )
#     parser.add_argument(
#             "--number_test", dest="number_test", type=int
#         )
    
    parser.set_defaults(
        model = "resnet18",
        numchar = 4,
        lr = 1e-3,
        epoch = 10
    )
    return parser.parse_args()

prog_args = arg_parse()
learning_rate = prog_args.lr
model_type = prog_args.model
num_epochs = prog_args.epoch
num_char = prog_args.numchar
image_width = 32
image_height = 32
batch_size = 256

trainset = Captcha('data/all/train',image_width,image_height, train=True)
testset = Captcha('data/all/test',image_width,image_height, train=False)

trainLoader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=2)
testLoader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=2)
                  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(string.ascii_uppercase + string.ascii_lowercase + string.digits)

if model_type == "resnet18":
    model = captcha_resnet.resnet18(num_classes = num_classes, num_char = num_char).to(device)
elif model_type == "resnet34":
    model = captcha_resnet.resnet34(num_classes = num_classes, num_char = num_char).to(device)
elif model_type == "resnet50":
    model = captcha_resnet.resnet50(num_classes = num_classes, num_char = num_char).to(device)
elif model_type == "wide_resnet":
    model = captcha_resnet.wide_resnet50_2(num_classes = num_classes, num_char = num_char).to(device)
else:
    print("No model")
                  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                  
count = 0
loss_list = []
test_loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for images, label in trainLoader:
        images, label = images.to(device), label.to(device)
        label = label.long()
        optimizer.zero_grad()
        
        output = model(images)
        loss = torch.sum(torch.stack([criterion(output[j], label[:,j]) for j in range(len(output))]))
        loss.backward()
        optimizer.step()
        count += 1
        
        if not (count % 50):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0
        
            for images, label in testLoader:
                images, label = images.to(device), label.to(device)
                label = label.long()
                output = model(images)
                test_loss = torch.sum(torch.stack([criterion(output[j], label[:,j]) for j in range(len(output))]))
                predictions = [torch.max(output[i], 1)[1].to(device) for i in range(len(output))]
                
                for c in range(len(output)):
                    correct += (predictions[c] == label[:,c]).sum()
            
                total += len(label)*len(output)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            test_loss_list.append(test_loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 50):
            print("Iteration: {}, Loss: {}, Test loss: {}, Accuracy: {}%".format(count, loss.data, test_loss.data, accuracy))
                  
SAVE_PATH = 'trained/'
SAVE_NAME = model_type

torch.save(model.state_dict(), SAVE_PATH + SAVE_NAME)
                  
import pandas as pd 
  
# dictionary of lists 
the_dict = {'iteration':  iteration_list,
        'loss': [a.cpu().detach().numpy() for a in loss_list],
        'test loss': [a.cpu().detach().numpy() for a in test_loss_list],
        'test Acc': [a.cpu().detach().numpy() for a in accuracy_list]}

df = pd.DataFrame(the_dict)                  
df.to_pickle( SAVE_PATH + SAVE_NAME + "_log.pkl")