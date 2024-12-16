import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import random
import matplotlib.pyplot as plt

def get_MNIST(type='iid', n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
    # idea for non-iid split from https://github.com/itslastonenikhil/federated-learning 
    # Download MNIST train and test dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    if type == "iid":
        print("Train Dataset")
        train = iid_split(train_dataset, n_clients, n_samples_train, batch_size, shuffle)
        print("Test Dataset")
        test = iid_split(test_dataset, n_clients, n_samples_test, batch_size, shuffle)
    elif type == "non_iid":
        print("Train Dataset")
        train = non_iid_split(train_dataset, n_clients, n_samples_train, batch_size, shuffle)
        print("Test Dataset")
        test = non_iid_split(test_dataset, n_clients, n_samples_test, batch_size, shuffle)
    else:
        train, test = [], []
    return train, test


def iid_split(dataset, nodes, samples_per_node, batch_size, shuffle):
    # print("Cstart split")
    loader = DataLoader(dataset, batch_size=samples_per_node, shuffle=shuffle)
    itr = iter(loader)
    data = []
    for i in range(nodes):
        node_dataloader = DataLoader(TensorDataset(*(next(itr))), batch_size=batch_size, shuffle=shuffle)
        data.append(node_dataloader)
    # print("working")
    return data


def non_iid_split(dataset, nodes, samples_per_node, batch_size, shuffle):
        # print("Cstart split")
    num_of_labels = len(dataset.classes)
    node_labels = []
    labels_per_node = int(num_of_labels * 0.3)
    random.seed(420)
    for _ in range(nodes):
        node_labels.append(random.sample(range(0, 9), labels_per_node))
    loader = DataLoader(dataset, batch_size=nodes * samples_per_node, shuffle=shuffle)
    itr = iter(loader)
    images, labels = next(itr)
    data = []
    for i in range(nodes):
        is_label_equal = [(node_label == labels) for node_label in node_labels[i]]
        index = torch.stack(is_label_equal).sum(0).bool()
        node_dataloader = DataLoader(TensorDataset(images[index], labels[index]), batch_size=batch_size, shuffle=shuffle)
        data.append(node_dataloader)
        # print("working")
    return data


# Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[128, 64, 64], num_classes=10):

        super(SimpleNN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # print(hidden_dims)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())  
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        return self.network(x)

# Client Class
class Client:
    def __init__(self, model, dataloader, lr=0.01, local_epochs=5, mu=1e4):

        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu  # fedprox regularization term

    def synchronize(self, global_state_dict):
        self.model.load_state_dict(global_state_dict)

    def train(self, global_state_dict):

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        global_params = {name: param.clone().detach() for name, param in global_state_dict.items()}

        for epoch in range(self.local_epochs):  
            for X, y in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(X)

                loss = loss_fn(outputs, y)

                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += torch.sum((param - global_params[name]) ** 2)
                loss += (self.mu / 2) * proximal_term  # fedbrox loss

                loss.backward()
                optimizer.step()

        return self.model.state_dict()


# Server Class
class Server:
    def __init__(self, model, clients):
        self.model = model
        self.clients = clients
        self.test_accuracies = []

    def aggregate(self, client_weights):
        """averaging client wieghts."""
        averaged_weights = {name: torch.zeros_like(param) for name, param in self.model.state_dict().items()}
        num_clients = len(client_weights)

        for weights in client_weights:
            for name, param in weights.items():
                averaged_weights[name] += param

        for name in averaged_weights:
            averaged_weights[name] /= num_clients

        return averaged_weights

    def federated_train(self, test_data, rounds=5):
        """running fedprox."""
        
        for r in range(rounds):
            print(f"Round {r + 1}/{rounds}")

            global_state_dict = self.model.state_dict()

            client_weights = []
            for client in self.clients:
                client.synchronize(global_state_dict)  
                client_weights.append(client.train(global_state_dict))  

            averaged_weights = self.aggregate(client_weights)

            self.model.load_state_dict(averaged_weights)

            test_accuracy = 0
            for dataloader in test_data:
                test_accuracy += server.test(dataloader)
            test_accuracy /= len(test_data)
            self.test_accuracies.append(test_accuracy)

    def test(self, dataloader):
        """Test the global model."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total


# Main Function
if __name__ == "__main__":
    for mu in [0, 0.0001, 0.1, 1, 5]:

        train_data, test_data = get_MNIST(type='iid', n_clients=5)

        input_dim = 784
        num_classes = 10
        global_model = SimpleNN(input_dim=input_dim, num_classes=num_classes)

        clients = []
        for dataloader in train_data:
            model = SimpleNN(input_dim=input_dim, num_classes=num_classes)
            clients.append(Client(model, dataloader, lr=0.1, local_epochs=10, mu = mu))

        server = Server(global_model, clients)

        server.federated_train(test_data=test_data, rounds=10)

        plt.plot(server.test_accuracies, label=f"mu ={mu}")
    plt.title("IID Data")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Rounds")
    plt.ylim((0,1))
    plt.legend()
    plt.show()

    for mu in [0, 0.0001, 0.1, 1, 5]:

        # Prepare Data
        train_data, test_data = get_MNIST(type='non_iid', n_clients=5)

        # Initialize Global Model
        input_dim = 784
        num_classes = 10
        global_model = SimpleNN(input_dim=input_dim, num_classes=num_classes)

        # Create Clients
        clients = []
        for dataloader in train_data:
            model = SimpleNN(input_dim=input_dim, num_classes=num_classes)
            clients.append(Client(model, dataloader, lr=0.1, local_epochs=10, mu = mu))

        # Create Server
        server = Server(global_model, clients)

        # Train Federated Model
        server.federated_train(test_data=test_data, rounds=10)

        plt.plot(server.test_accuracies, label=f"mu ={mu}")
    plt.title("Non-IID Data")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Rounds")
    plt.ylim((0,1))
    plt.legend()
    plt.show()