import gradio as gr
import torch
import torchvision
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Class names
class_names = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware ice/snow", "Wild animals crossing",
    "End restrictions", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End no passing", "End no passing >3.5t"
]

# Model yükle
weights = torchvision.models.ViT_B_16_Weights.DEFAULT
model = torchvision.models.vit_b_16(weights=None)
model.heads = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),
    torch.nn.Linear(768, len(class_names))
)
model.load_state_dict(torch.load("model/gtsrb_vit_4epoch.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sınıf açıklamaları
class_descriptions = {
    "Speed limit 20": "Maximum speed allowed: 20 km/h",
    "Speed limit 30": "Maximum speed allowed: 30 km/h",
    "Speed limit 50": "Maximum speed allowed: 50 km/h",
    "Speed limit 60": "Maximum speed allowed: 60 km/h",
    "Speed limit 70": "Maximum speed allowed: 70 km/h",
    "Speed limit 80": "Maximum speed allowed: 80 km/h",
    "End speed limit 80": "End of 80 km/h speed restriction",
    "Speed limit 100": "Maximum speed allowed: 100 km/h",
    "Speed limit 120": "Maximum speed allowed: 120 km/h",
    "No passing": "Overtaking prohibited for all vehicles",
    "No passing >3.5t": "Overtaking prohibited for vehicles over 3.5 tons",
    "Right-of-way": "You have right of way at the next intersection",
    "Priority road": "You are on a priority road",
    "Yield": "Give way to oncoming traffic",
    "Stop": "Come to a complete stop before proceeding",
    "No vehicles": "No vehicles allowed",
    "No vehicles >3.5t": "No vehicles over 3.5 tons allowed",
    "No entry": "Entry prohibited for all vehicles",
    "General caution": "General hazard ahead — proceed with caution",
    "Dangerous curve left": "Sharp curve to the left ahead",
    "Dangerous curve right": "Sharp curve to the right ahead",
    "Double curve": "Double bend ahead",
    "Bumpy road": "Uneven road surface ahead",
    "Slippery road": "Slippery road conditions ahead",
    "Road narrows right": "Road narrows on the right side",
    "Road work": "Road works ahead — slow down",
    "Traffic signals": "Traffic lights ahead",
    "Pedestrians": "Pedestrian crossing ahead",
    "Children crossing": "Children crossing — school zone ahead",
    "Bicycles crossing": "Bicycle crossing ahead",
    "Beware ice/snow": "Icy or snowy road conditions ahead",
    "Wild animals crossing": "Wild animals may cross the road",
    "End restrictions": "End of all restrictions",
    "Turn right ahead": "Turn right at the next intersection",
    "Turn left ahead": "Turn left at the next intersection",
    "Ahead only": "Straight ahead only — no turning",
    "Go straight or right": "Go straight or turn right",
    "Go straight or left": "Go straight or turn left",
    "Keep right": "Keep to the right side of the road",
    "Keep left": "Keep to the left side of the road",
    "Roundabout mandatory": "Enter the roundabout",
    "End no passing": "End of no overtaking zone",
    "End no passing >3.5t": "End of no overtaking zone for vehicles over 3.5 tons"
}


def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    top5 = torch.topk(probs, 5)
    results = {class_names[i]: float(probs[i]) for i in top5.indices}

    top_class = class_names[top5.indices[0]]
    description = class_descriptions.get(top_class, "")

    return results, description


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Traffic Sign"),
    outputs=[
        gr.Label(num_top_classes=5, label="Prediction"),
        gr.Textbox(label="Description")
    ],
    title="🚦 Traffic Sign Classifier",
    description="Upload a traffic sign image. Fine-tuned ViT-B/16 on GTSRB dataset — 43 classes, 85% test accuracy.",
    theme=gr.themes.Soft(),
    flagging_mode="never"
)

demo.launch()