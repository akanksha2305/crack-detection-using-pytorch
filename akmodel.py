import os 
import io 
import torch #deep learning
import torchvision #datasets and transformations for computer vision
import torchvision.transforms as transforms #image transformations
import torch.nn as nn  #layers used to build nn
import torch.optim as optim #optimization algorithms
import torch.nn.functional as F #activation functions
from PIL import Image
import multiprocessing
import matplotlib.pyplot as plt
import time



MODEL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'crack_detection_pytorch/data')

train_dir = os.path.join(MODEL_DATA_PATH, 'train')
validation_dir = os.path.join(MODEL_DATA_PATH, 'validation')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(14400, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 14400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model():
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    trainset = torchvision.datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    testset = torchvision.datasets.ImageFolder(
        root=validation_dir,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    accuracy_list=[]
    loss_list=[]
    start_time = time.time()

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Calculate accuracy on test set after each epoch
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        loss_list.append(running_loss / len(trainloader))
        print('Accuracy of the network on the %d test images: %d %%' % (total, accuracy))
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Training completed in {training_time} seconds.')
    plt.plot(range(1, 6), accuracy_list, label='Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    plt.plot(range(1, 6), loss_list, label='Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), 'crack_detection.pth'))

def predict_image_class(img):
    # Load the image and apply transformations
    img = Image.open(io.BytesIO(img)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    tensor = transform(img).unsqueeze(0)

    # Load the saved model and make a prediction
    net = Net()
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'crack_detection.pth')))
    net.eval()
    outputs = net(tensor)
    _, y_hat = torch.max(outputs, 1)

    # Map the predicted class to the corresponding label
    prediction_classes = ('crack', 'no_crack')
    prediction = prediction_classes[y_hat]

    print(prediction)
def predict():
    MODEL_PATH = "C:\\Users\\akank\\OneDrive\\Documents\\projects\\crack_detection_pytorch\\crack_detection.pth"

    # Load trained model
    net = Net()
    net.load_state_dict(torch.load(MODEL_PATH))
    print('Model loaded. Start serving...')
    
    # Load test image as bytes
    with open("test11.jpg", "rb") as f:
        img_bytes = f.read()
    
    # Call the prediction function with the image bytes
    predict_image_class(img_bytes)

if __name__ == "__main__":  
    multiprocessing.freeze_support() 
    #train_model()
    predict()
