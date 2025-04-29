import torch
from torch import nn
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models import DeepONet, BelNet, StochasticFeatures
from utils import get_data, train_model, validate_model

'''
    prepare data for deeponet
'''
np.random.seed(42) # fixed seed
train_data, test_data = get_data(1000,200)
train_data= [torch.from_numpy(d).float().to(device).reshape((1000,128,1)) for d in train_data]
test_data = [torch.from_numpy(d).float().to(device).reshape((200,128,1)) for d in test_data]
(x_train, a_train, y_train, u_train) = train_data
(x_test, a_test, y_test, u_test)     = test_data
print("train shape:", x_train.shape, a_train.shape, y_train.shape, u_train.shape)
print("test shape:", x_test.shape, a_test.shape, y_test.shape, u_test.shape)
    

np.random.seed(42)
torch.manual_seed(42)

example_num = 5
print(f"using examples {example_num}")
if example_num == 0:
    model = DeepONet().to(device)
elif example_num == 1:
    model = BelNet(num_learned_basis = 50, num_fixed_basis = 0)
elif example_num == 2:
    basis = StochasticFeatures(kernel="trig", x_dim=1, num_features=50, prior = torch.nn.init.normal_)
    model = BelNet(num_learned_basis = 50, num_fixed_basis = 50, basis = basis)
elif example_num == 3:
    prior = lambda w: torch.nn.init.uniform(w, a = 0.1, b=10)
    basis = StochasticFeatures(kernel="trig", x_dim=1, num_features=50, prior = prior)
    model = BelNet(num_learned_basis = 50, num_fixed_basis = 50, basis = basis)
elif example_num == 4:
    def prior(w):
        with torch.no_grad():
            w.copy_(torch.linspace(0.1, 10, w.shape[0]).view(w.shape[0], 1))
    basis = StochasticFeatures(kernel="trig", x_dim=1, num_features=50, prior = prior)
    model = BelNet(num_learned_basis = 50, num_fixed_basis = 50, basis = basis)
model = model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model has {count_parameters(model)} parameters")
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_losses = train_model(model, train_data, test_data, criterion, optimizer, num_epochs=500000, device=device)
validate_model(model, test_data, criterion, device=device)

# save model
torch.save(model.state_dict(), 'model_save.pth')
