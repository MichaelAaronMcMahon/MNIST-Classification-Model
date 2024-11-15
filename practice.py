import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from random import randrange
import math

train_dataset = datasets.MNIST(root='data/', train=True, transform=None, download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=None, download=True)

train_in_np = np.ndarray((60000, 28, 28))
train_out_np = np.zeros(60000)
test_in_np = np.ndarray((60000, 28, 28))
test_out_np = np.zeros(60000)
for i in range(60000):
    train_in_np[i] = train_dataset[i][0]
    train_out_np[i] = train_dataset[i][1]

for i in range(10000):
    test_in_np[i] = test_dataset[i][0]
    test_out_np[i] = test_dataset[i][1]

train_in_torch = torch.from_numpy(train_in_np)
train_out_torch = torch.from_numpy(train_out_np)
test_in_torch = torch.from_numpy(test_in_np)
test_out_torch = torch.from_numpy(test_out_np)

class SimpleMNISTSoftMax(nn.Module):
    def __init__(self):
        super(SimpleMNISTSoftMax, self).__init__()
        self.weight_matrix = torch.nn.Parameter(torch.randn(10, 28*28, dtype=torch.float64), requires_grad = True)
        self.bias_vector = torch.nn.Parameter( torch.randn(10, 1, dtype=torch.float64), requires_grad = True)
    def forward(self, input_tensor):
        flattened = nn.Flatten()(input_tensor)
        linear_transforamtion = torch.matmul(self.weight_matrix, flattened.t()) + self.bias_vector
        linear_transforamtion = linear_transforamtion.t()
        final_probabilities = nn.Softmax(dim=1)(linear_transforamtion)
        return linear_transforamtion, final_probabilities

model = SimpleMNISTSoftMax()
y_pred = model(train_in_torch)

def confusion_matrix(model, x, y, dp):
  identification_counts = np.zeros( shape = (10,10), dtype = np.int32 )
  probabilities = model( x )[1] # Compute all the probabilities for each data point in x
  predicted_classes = torch.argmax( probabilities, dim = 1 ) # For each set of probabilities, identify the most likely class
  for i in range(dp):
    actual_class = int(y[i])
    predicted_class = predicted_classes[i].item()
    identification_counts[actual_class, predicted_class] += 1 # Tally that something of actual_class was most likely (by the model) to be predicted_class
  return identification_counts

def get_batch( x, y, batch_size ):
  n = x.shape[0]
  batch_indices = random.sample( [ i for i in range(n) ], k = batch_size )
  x_batch = x[ batch_indices ]
  y_batch = y[ batch_indices ]
  return x_batch, y_batch.to(torch.int64)

print("Training data:")
print( confusion_matrix( model, train_in_torch, train_out_torch, 60000 ) )

optimizer = optim.Adam( model.parameters() )
loss_function = nn.CrossEntropyLoss()
batch_size = 50

for epochs in range(15):
  total_loss = 0
  for batch in range( 60000 // batch_size ):
    x_batch, y_batch = get_batch( train_in_torch, train_out_torch, batch_size )
    optimizer.zero_grad() # This automatically zeros the gradients of the specified parameters, so we don't have to.
    logits, probabilities = model( x_batch )
    loss = loss_function( logits, y_batch )
    total_loss += loss.item()
    loss.backward()
    optimizer.step()

  print("Average Loss per Data Point:", total_loss / ( 60000 // batch_size ) )

print("Training data:")
print( confusion_matrix( model, train_in_torch, train_out_torch, 60000 ) )

print("Testing data:")
print(confusion_matrix(model, test_in_torch, test_out_torch, 10000))