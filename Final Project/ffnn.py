import torch
import torch.nn as nn
from random import sample
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import csv


from helpers import plot_decision_regions, process_attribute, process_height

def read_data_ffnn(file_name, include_undrafted, undrafted_round, round_specific=True):
    '''
    read_data_ffnn takes in a name of a csv file, a boolean for whether or not to include undrafted players, and a "round" number
    for undrafted players. Returns three numpy arrays: x_data, y_data, and labels.
    x_data: a numpy array of shape (n_samples, 3) where n_samples is the number of samples in the dataset. Index 0 is the
    bias term, and indexes 1 and 2 are the two input features.
    y_data: a numpy array of shape (n_samples, 1) where n_samples is the number of samples in the dataset. Index 0 is the
    output label.
    labels: a numpy array of shape (n_samples, 2) where n_samples is the number of samples in te dataset. Indexes 0 and 1
    are 1 depending on the output label. This is a one-hot encoded version of y_data.
    '''
    x_data = []
    y_data = []
    label_data = []
    
    # Get all positions of players
    positions_set = set()
    
    # Other attributes from combine
    height_list = []
    weight_list = []
    yd40_list = []
    vertical_list = []
    bench_list = []
    
    with open(file_name, 'rt') as file:
        # Map every row to dictionaries using the header row as keys
        reader = csv.DictReader(file)        
        # Iterate over all data
        for row in reader:
            positions_set.add(row['Pos'])
            height_list.append(process_height(row['Height']))
            weight_list.append(process_attribute(row['Weight']))
            yd40_list.append(process_attribute(row['40yd']))
            vertical_list.append(process_attribute(row['Vertical']))
            bench_list.append(process_attribute(row['Bench']))
    
    # Find the min and max values for position, height, weight, yd40, vertical, and bench to normalize values
    height_min = min([val for val in height_list if val > 0])
    height_max = max(height_list)
    height_mean = sum(height_list) / len(height_list)
    weight_min = min([val for val in weight_list if val > 0])
    weight_max = max(weight_list)
    weight_mean = sum(weight_list) / len(weight_list)
    yd40_min = min([val for val in yd40_list if val > 0])
    yd40_max = max(yd40_list)
    yd40_mean = sum(yd40_list) / len(yd40_list)
    vertical_min = min([val for val in vertical_list if val > 0])
    vertical_max = max(vertical_list)
    vertical_mean = sum(vertical_list) / len(vertical_list)
    bench_min = min([val for val in bench_list if val > 0])
    bench_max = max(bench_list)
    bench_mean = sum(bench_list) / len(bench_list)

    
    # Assign each position a location in the vector for positions
    positions_dict = {pos: idx for idx, pos in enumerate(positions_set)}
    
    with open(file_name,'rt') as file:       
        # Map every row to dictionaries using the header row as keys
        reader = csv.DictReader(file)
        # Iterate over every row to add each player to the data_set list
        for row in reader:            
            
            # Positions list initialization
            positions_list = [0] * len(positions_set)
            # Set the position for the given player
            positions_list[positions_dict[row['Pos']]] = 10
            
            height = process_height(row['Height'])
            if height != -1:
                # Normalize the height
                height = (height - height_min) / (height_max - height_min)
            else:
                height = height_mean
            # Convert weight into float value
            weight = process_attribute(row['Weight'])
            if weight != -1:
                # Normalize the weight
                weight = (weight - weight_min) / (weight_max - weight_min)
            else:
                weight = weight_mean
            # Convert 40yd into float value
            yd40 = process_attribute(row['40yd'])
            if yd40 != -1:
                # Normalize the 40 yard dash
                yd40 = (yd40 - yd40_min) / (yd40_max - yd40_min)
            else:
                yd40 = yd40_mean
            # Convert Vertical into float value
            vertical = process_attribute(row['Vertical'])
            if vertical != -1:
                # Normalize the vertical
                vertical = (vertical - vertical_min) / (vertical_max - vertical_min)
            else:
                vertical = vertical_mean
            # Convert bench into float value
            bench = process_attribute(row['Bench'])
            if bench != -1:
                # Normalize the bench
                bench = (bench - bench_min) / (bench_max - bench_min)
            else:
                bench = bench_mean
            
            if include_undrafted:
                x_data.append([1.0] + positions_list + [height,
                                                        weight,
                                                        yd40, 
                                                        vertical, 
                                                        bench])
                if row['Round'] == "":
                    round = str(undrafted_round - 1)
                else:
                    round = str(int(row['Round']) - 1)
                y_data.append([round])
                temp = [0] * 8
                temp[int(round) - 1] = 1
                label_data.append(temp)
            else:
                if row['Round'] != "":
                    x_data.append([1.0] + positions_list + [height,
                                                            weight,
                                                            yd40, 
                                                            vertical, 
                                                            bench])
                    if not round_specific:
                        if int(row['Round']) in [1, 2, 3]:
                            round = '0'
                        # elif int(row['Round']) in [3, 4, 5]:
                        #     round = '1'
                        else:
                            round = '1'
                    else:
                        round = str(int(row['Round']) - 1)
                    y_data.append([round])
                    temp = [0] * 8
                    temp[int(round) - 1] = 1
                    label_data.append(temp)
                
            
    xs = np.array(x_data, dtype='float32')
    ys = np.array(y_data, dtype='float32')
    labels = np.array(label_data, dtype='float32')
    
    return (xs, ys, labels)

################################
# Split Data
################################
def split_data_ffnn(dataset):
    '''
    Takes in data from read_data_ffnn and returns three datasets: a training set,
    validation set, and a test set. The size of each set is selected by hyperparameters.
    '''
    
    data = dataset[0]
    
    # Hyperparameters for dataset sizes
    train_size = int(len(data) // 3)
    valid_size = int(len(data) // 3)
    test_size = len(data) - train_size - valid_size
    
    if len(data) != len(data) or len(data) != len(data):
        raise("Data lengths are not all the same")
    
    # Randomized list of data
    indices = sample(range(len(data)), len(data))
        
    train_set = dataset[0][indices][:train_size], dataset[1][indices][:train_size], dataset[2][indices][:train_size]
    valid_set = dataset[0][indices][train_size:train_size+valid_size], dataset[1][indices][train_size:train_size+valid_size], dataset[2][indices][train_size:train_size+valid_size], 
    test_set = dataset[0][indices][train_size+valid_size:], dataset[1][indices][train_size+valid_size:], dataset[2][indices][train_size+valid_size:]
    
    return train_set, valid_set, test_set

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.x_data, self.y_data, self.labels = data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y


# Define the neural network
class net3(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, activation):
        super(net3, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.l2 = nn.Linear(hidden_size, output_size)  # First hidden to output
        # self.l2 = nn.Linear(hidden_size, hidden_size)  # First hidden to second hidden
        # self.l3 = nn.Linear(hidden_size, output_size)  # Second hidden to third hidden
        self.activation = activation  # define the activation function

    def forward(self, x):
        output = self.l1(x)
        output = self.activation(output)
        output = self.l2(output)
        return output


# training function
def train(model, data_loader, loss_function, optimizer):

    model.train()
    train_loss = 0  # for plotting learning curves for training and validation loss as a function of training epochs

    for batch, (X, y) in enumerate(data_loader):

        pred = model(X)
        loss = loss_function(pred,
                             y.squeeze().long())  # Originally y.unsqueeze(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(data_loader)
    return train_loss


# testing function
def test(model, data_loader, loss_function):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X)
            loss = loss_function(
                pred,
                y.squeeze().long()).item()  # Originally y.unsqueeze(1)
            test_loss += loss

    test_loss /= len(data_loader)
    return test_loss


# output the final accuracy of the model
def final_accuracy(model, data_loader):

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for X, y in data_loader:   
            pred = model(X)
            pred = torch.round(pred)
            pred_label = np.argmax(pred.numpy(), axis=1).astype(float)
            # y_label = y.squeeze().numpy()
            if y.numel() == 1:
                y_label = np.array([y.item()])
            else:
                y_label = y.squeeze().numpy()
            print(y_label)
            print(y_label.shape)
            print(pred_label)
            print(pred_label.shape)
            for idx, _ in enumerate(pred_label):
                total_predictions += 1
                correct_predictions += (pred_label[idx] == y_label[idx])

    return correct_predictions / total_predictions


def plot_training_and_validation_loss(train_loss, test_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,
                   len(train_loss) + 1),
             train_loss,
             label="Training Loss",
             marker='o')
    plt.plot(range(1,
                   len(test_loss) + 1),
             test_loss,
             label="Validation Loss",
             linestyle="--",
             marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(
        "Learning curves for training and validation loss as a function of training epochs"
    )
    plt.legend()
    plt.grid(True)
    plt.ion()
    plt.show()


if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(0)

    round_specific = False

    # hyperparameters
    input_size = 28
    if round_specific:
        output_size = 8
    else:
        output_size = 2
    learning_rate = 0.002
    num_epochs = 100
    batch_size = 10
    undrafted_round = 8
    include_undrafted = False

    # Part 1
    # activation_function = nn.ReLU()
    loss_function = nn.CrossEntropyLoss()

    # Part 2
    activation_function = nn.Sigmoid()
    # loss_function = nn.MSELoss()

    # possible values
    hidden_sizes = [40, 60, 80]

    # iterate through each name and hidden size
    for hidden_size in hidden_sizes:

        # initialize loss lists
        train_loss = []
        test_loss = []
        
        combine_data = read_data_ffnn('data/nfl_combine_2010_to_2023.csv', include_undrafted, undrafted_round, round_specific)
        train_data, valid_data, test_data = split_data_ffnn(combine_data)

        # define the datasets and the loaders
        train_dataset = CustomDataset(train_data)
        valid_dataset = CustomDataset(valid_data)
        test_dataset = CustomDataset(test_data)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        # build the model and optimizer
        model = net3(input_size, hidden_size, output_size, activation_function)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train the model
        for epoch in range(num_epochs):
            train_losses = train(model, train_loader, loss_function, optimizer)
            test_losses = test(model, valid_loader, loss_function)
            train_loss.append(train_losses)
            test_loss.append(test_losses)

        # show the final accuracy of the model
        print('Final accuracy on test set (k = %i): %.2f' %
              (hidden_size, final_accuracy(model, test_loader)))

        # plot the train loss and test loss against the number of epochs in seperate graphs
        plot_training_and_validation_loss(train_loss, test_loss)

        # x_vals, targets, _ = read_data_ffnn('data/nfl_combine_2010_to_2023.csv', include_undrafted, undrafted_round, round_specific)

        # features = np.array([inner_list[1:] for inner_list in x_vals])
        # print(x_vals)
        # print(targets)
        # print(features)
        # plot_decision_regions(features, targets, model)

        input("Press enter to continue")
