import torch
import torch.optim as optim
import torch.nn as nn
from chest_dataset import load_data
from chest_xray import ChestXrayNet

train_dir = './chest_xray_data/train'
test_dir = './chest_xray_data/test'

train_loader, test_loader = load_data(train_dir, test_dir)

model = ChestXrayNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Epoch 50/50, Loss: 0.013127676149856092
# Accuracy on test set: 91.89%

torch.save(model.state_dict(), "chest_xray_weights.pt")
