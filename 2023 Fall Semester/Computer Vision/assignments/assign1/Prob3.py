import os
import cv2
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Prob2 import CNN_Model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ClockDataset(Dataset):
    def __init__(self, root_path, image_folder, json_file, transform=None):

        json_path = os.path.join(root_path, json_file)
        image_folder_path = os.path.join(root_path, image_folder)
        
        self.transform = transform                
        # read json file
        with open(json_path, "r") as f:
            self.data_info = json.load(f)
        
        # extract image paths, hours, and minutes from the data
        self.image_paths = [os.path.join(image_folder_path, item['image']) for item in self.data_info]
        self.hours = [item['hour'] - 1 for item in self.data_info] # convert to categorical labels (0-11)
        self.minutes = [item['minute'] for item in self.data_info] # convert to categorical labels 

    def __len__(self):
        return  len(self.data_info)
 
    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        hour = torch.tensor(self.hours[idx], dtype=torch.long)
        minute = torch.tensor(self.minutes[idx], dtype=torch.long)
        
        # Read and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)

        return img, hour, minute
    

root_path = "./dataset"
image_folder = "clock_images"
json_file = "data_info.json"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet mean and std (we use this because we use a pretrained model)
])


# Hyperparameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
BATCH_SIZE = 32

# load dataset
dataset = ClockDataset(root_path, image_folder, json_file, transform=transform)

# split dataset into train and test sets
train_set, test_set = torch.utils.data.random_split(dataset, [2400, 600])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
model = CNN_Model()
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train function
def train(model, train_dataloader, optimizer, criterion, device, NUM_EPOCHS):
    
    # train mode
    model.train()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        
        total_epoch_loss = 0
        total_samples = 0
        correct_hour = 0
        correct_minute = 0

        for images, hours, minutes in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device)
            
            optimizer.zero_grad()

            hour_pred, minute_pred = model(images)
            
            loss_hour = criterion(hour_pred, hours)
            loss_minute = criterion(minute_pred, minutes)

            loss = loss_hour + loss_minute

            loss.backward()
            optimizer.step()

            # calculate accuracy
            
            with torch.no_grad():
                # get predictions
                _, predicted_hour = torch.max(hour_pred.data, 1)
                _, predicted_minute = torch.max(minute_pred.data, 1)
                
                # update counts
                correct_hour += (predicted_hour == hours).sum().item()
                correct_minute += (predicted_minute == minutes).sum().item()

                total_samples += images.size(0)
                total_epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            
            # report accuracy
            accuracy_hour = correct_hour / total_samples
            accuracy_minute = correct_minute / total_samples


            print("-"*50)
            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {total_epoch_loss}, '
                    f'Train Accuracy (Hour): {accuracy_hour * 100:.2f}%, '
                    f'Train Accuracy (Minute): {accuracy_minute * 100:.2f}%')
            print("-"*50)
            
            # save model checkpoint
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")


# test function
def test(model, test_dataloader):
    
    model.eval()
    test_samples = 0
    correct_hour_test = 0
    correct_minute_test = 0

    with torch.no_grad():
        for images, hours, minutes in test_dataloader:
            images, hours, minutes = images.to(device), hours.to(device), minutes.to(device)

            hour_pred, minute_pred = model(images)

            _, predicted_hour = torch.max(hour_pred.data, 1)
            _, predicted_minute = torch.max(minute_pred.data, 1)

            test_samples += images.size(0)
            correct_hour_test += (predicted_hour == hours).sum().item()
            correct_minute_test += (predicted_minute == minutes).sum().item()

    accuracy_hour_test = correct_hour_test / test_samples
    accuracy_minute_test = correct_minute_test / test_samples

    print(f'Test Accuracy (Hour): {accuracy_hour_test * 100:.2f}%, '
          f'Test Accuracy (Minute): {accuracy_minute_test * 100:.2f}%')
    
    
if __name__ == "__main__":
    
    train(model, train_loader, optimizer, criterion, device, NUM_EPOCHS)
    test(model, test_loader)