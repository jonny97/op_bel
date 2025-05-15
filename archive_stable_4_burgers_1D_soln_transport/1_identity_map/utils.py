'''
    fully mimic lulu with new data 
'''
import numpy as np
from scipy import io
import torch

def get_data(ntrain, ntest):
    # input (x, a, y)
    # output (u)
    N = ntrain + ntest
    data = io.loadmat("/workspace/dataGRFshorttspan.mat")

    def get_input_indices(N):
        indices_1 = np.arange(0, 2**13, 2**6)
        index_array = np.zeros((N, 2**7), dtype=int) 
        for n in range(N):
            index_array[n] = indices_1
        return index_array
    
    def get_output_indices(N):
        indices_1 = np.arange(0, 2**13, 2**6)
        index_array = np.zeros((N, 2**7), dtype=int) 
        for n in range(N):
            index_array[n] = indices_1
        return index_array

    input_index_array = get_input_indices(N)
    output_index_array = get_output_indices(N)

    input_func_data  = data["output"][:N, :].astype(np.float32)
    output_func_data = data["output"][:N, :].astype(np.float32)

    x_data = input_index_array * (1.0 / 2**13)
    a_data = input_func_data[np.arange(N)[:, None], input_index_array]

    y_data = output_index_array * (1.0 / 2**13)
    u_data = output_func_data[np.arange(N)[:, None], output_index_array]

    train_data = (x_data[:ntrain], a_data[:ntrain], y_data[:ntrain], u_data[:ntrain]) 
    test_data  = (x_data[ntrain:], a_data[ntrain:], y_data[ntrain:], u_data[ntrain:])

    return train_data, test_data

def inverse_time_decay(epoch, initial_lr, decay_factor, decay_epochs):
    """Inverse time decay function."""
    return initial_lr / (1 + decay_factor * (epoch / decay_epochs))

# Training function
def train_model(model, train_data, test_data, criterion, optimizer, num_epochs, device):
    x_train, a_train, y_train, u_train = train_data
    model.train()  # Set the model to training mode
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = inverse_time_decay(epoch, initial_lr=0.001, decay_factor=0.5, decay_epochs=100000)
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(x_train, a_train, y_train)
        loss = criterion(outputs, u_train)
        loss.backward()
        optimizer.step()
        
        epoch_loss = loss.item()
        train_losses.append(epoch_loss)
        if epoch % 5000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            validate_model(model, test_data, criterion, device=device)
    return train_losses

def validate_model(model, test_data, criterion, device):
    x_test, a_test, y_test, u_test = test_data
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    rel_l2_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation during validation
        actual_batch_size = u_test.shape[0]
        outputs = model(x_test, a_test, y_test)
        loss = criterion(outputs, u_test)
        val_loss += loss.item()
        diff_norms = torch.norm(outputs.reshape(actual_batch_size,-1) - u_test.reshape(actual_batch_size,-1), dim=1)
        y_norms = torch.norm(u_test.reshape(actual_batch_size,-1), dim=1)
        rel_l2_loss += torch.mean(diff_norms/y_norms)   
    
    print(f"Validation Loss: {val_loss:.5f}, mean Relative L2 Loss: {rel_l2_loss:.5f}")