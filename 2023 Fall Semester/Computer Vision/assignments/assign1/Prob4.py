import cv2
import torch
from Prob2 import CNN_Model
from torchvision import transforms

   
def test_model(model, image_path, transform=None):
   
    model.eval()

    with torch.no_grad():
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if transform is not None:
            image = transform(image)
        
        image = image.unsqueeze(0).to(device) # add batch dimension and move to GPU if available

        hour_pred, minute_pred = model(image)

        _, predicted_hour = torch.max(hour_pred.data, 1)
        _, predicted_minute = torch.max(minute_pred.data, 1)

        predicted_hour = predicted_hour.item() + 1 # convert back to 1-12

    return predicted_hour, predicted_minute.item()

# load model

model = CNN_Model()
model.load_state_dict(torch.load("checkpoints/Prob3.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # ImageNet mean and std (we use this because we use a pretrained model)
])

image_path = "provided_clock_examples/3.jpg" # example single image path

predicted_hour, predicted_minute = test_model(model, image_path, transform)

print(f"Predicted Hour: {predicted_hour}")
print(f"Predicted Minute: {predicted_minute}")

