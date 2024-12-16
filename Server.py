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
