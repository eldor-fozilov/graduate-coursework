
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from model import MyModel
from dataset import FoodDataset
from train_and_test_utils import train, test # , train_transform


# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4


# Data path (train and test)
train_data_path = "train.pkl"
val_data_path = "val.pkl"


# transforms.RandomChoice([transforms.RandomRotation((-45, 45)), transforms.RandomAffine(degrees=(-45,45), translate=(0.1, 0.3))]),

train_transform = transforms.Compose([
    transforms.CenterCrop((512, 512)), # max size of the image
    transforms.Resize((256, 256)),
    transforms.RandomRotation((-45, 45)),
    # transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

test_transform = transforms.Compose(
    [   transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Load the data
train_dataset = FoodDataset(train_data_path, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = FoodDataset(val_data_path, transform=test_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# set up the model
model = MyModel()
model = model.to(device)

# set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# dummy_input = torch.rand(BATCH_SIZE, 256, 256, 3).to(device)
# dummy_output = model(dummy_input)
# summary(model, (3, 256,256))

for epoch in range(NUM_EPOCHS):
    ###Train Phase

    # Initialize Loss and Accuracy
    train_loss = 0.0
    train_accu = 0.0
    

    # Iterate over the train_dataloader
    for sample in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):

        curr_loss, num_correct = train(model, optimizer, sample)
        train_loss += curr_loss / len(train_loader)
        train_accu += num_correct / len(train_dataset)

    ### Test Phase
    # Initialize Loss and Accuracy
    test_loss = 0.0
    test_accu = 0.0

    # Iterate over the test_dataloader
    for sample in val_loader:
        curr_loss, num_correct = test(model, sample)
        test_loss += curr_loss / len(val_loader)
        test_accu += num_correct / len(val_dataset)
    
        
    checkpoint = {'model': MyModel(), 'model_state_dict': model.state_dict()}
    torch.save(checkpoint, f'./checkpoints/efficientnet_v2/model_after_{epoch + 1}_epochs.pth')
    print('[EPOCH {}] TR LOSS : {:.03f}, TE LOSS :{:.03f}, TR ACCU: {:.03f}, TE ACCU :{:.03f}'.format(epoch + 1, train_loss, test_loss, train_accu, test_accu))
