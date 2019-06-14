
# Homecoming (eYRC-2018): Task 1A
# Build a Fully Connected 2-Layer Neural Network to Classify Digits

# NOTE: You can only use Tensor API of PyTorch

from nnet import model

# TODO: import torch and torchvision libraries
# We will use torchvision's transforms and datasets
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# TODO: Defining torchvision transforms for preprocessing
# TODO: Using torchvision datasets to load MNIST
# TODO: Use torch.utils.data.DataLoader to create loaders for train and test
# NOTE: Use training batch size = 4 in train data loader.

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

train_set = dset.MNIST(root='./data', train=True, transform=trans, download=True)
test_set = dset.MNIST(root='./data', train=False, transform=trans, download=True)

batch_size=4
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)





# NOTE: Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: Don't change these settings
# Layer size
N_in = 28 * 28 # Input size
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.001


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# TODO: Define number of epochs
N_epoch = 7 # Or keep it as is


# TODO: Training and Validation Loop
# >>> for n epochs
## >>> for all mini batches
### >>> net.train(...)
## at the end of each training epoch
## >>> net.eval(...)



for epoch in range(N_epoch):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs=inputs.view(-1,N_in)   
        net.train(inputs,labels,lr)
print("TRAINING FINISHED")

# TODO: End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)
S=0
K=0
for batch_idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.view(-1, N_in)
    score, idx = net.predict(inputs)
    print("Score : ", score)
    print("Index : ", idx,"Labels : ",labels)
    k, s, creloss, accuracy, outputs=net.eval(inputs, labels, debug=True)
    S+=s
    K+=k
print("correct",K)
print("total", S)
print("net accuracy:", K/S)
print("Verify the results")
