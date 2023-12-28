import typing as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import DEVICE, synthesize_data, score_iou
from torchvision import models
from torchvision import transforms
from PIL import Image
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from torch.utils.data import DataLoader, TensorDataset
from efficientnet_pytorch import EfficientNet
class StarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 32, 3)
        self.bn0 = nn.BatchNorm2d(32)
        
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = nn.Conv2d(512, 512, 3)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn0(self.conv0(x))))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        regression_output = F.relu(self.fc4(x))
        classification_output = torch.sigmoid(self.classifier(x)).squeeze()
        return classification_output, regression_output


# class CustomEfficientNet(nn.Module):
#     def __init__(self, num_classes=1, num_bbox_outputs=4, num_yaw_outputs=1):
#         super().__init__()
#         # Load pre-trained EfficientNet
#         self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

#         # Get the feature size of EfficientNet (depends on the variant used)
#         feature_size = self.efficientnet._fc.in_features

#         # Replace the classifier with custom heads
#         self.classifier_head = nn.Linear(feature_size, num_classes)
#         self.regressor_head = nn.Linear(feature_size, num_bbox_outputs)
#         self.yaw_head = nn.Linear(feature_size, num_yaw_outputs)

#     def forward(self, x):
#         # EfficientNet features
#         x = self.efficientnet.extract_features(x)
#         x = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

#         # Pass the features through the custom heads
#         classification_output = self.classifier_head(x)
#         bbox_output = self.regressor_head(x)
#         yaw_output = self.yaw_head(x)

#         # Concatenate bbox and yaw predictions
#         combined_output = torch.cat((bbox_output, yaw_output), dim=1)

#         return classification_output, combined_output
class StarDataset(torch.utils.data.Dataset):
    """Return star image and labels"""

    def __init__(self, data_size=50000, preprocess = None):
        self.data_size = data_size
        self.preprocess = preprocess
    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image, label = synthesize_data()
        # image = image.astype(np.uint8)
        # image = Image.fromarray((image * 255).astype('uint8'), mode='L') #convert to gray scale
        # if self.preprocess:
        #     image = self.preprocess(image) 
        # label = torch.from_numpy(label).float()
        return image[None], label

# class StarMobileNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mobilenet = models.mobilenet_v2(pretrained=True).features
#         # # Freeze the parameters of MobileNet
#         # for param in self.mobilenet.parameters():
#         #     param.requires_grad = False

#         # Add an Adaptive Average Pool to reduce the spatial dimensions to 1x1
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         # Define the classifier, regressor, and yaw prediction heads
#         self.classifier_head = nn.Linear(1280, 1)  # For binary classification (star/no star)
#         self.regressor_head = nn.Linear(1280, 4)  # For bounding box (x, y, width, height)
#         self.yaw_head = nn.Linear(1280, 1)  # For yaw prediction

#     def forward(self, x):
#         # Get the features from the MobileNet
#         features = self.mobilenet(x)
#         features = self.pool(features)
#         features = features.view(features.size(0), -1)  # Flatten the features
#         # Apply the classifier and regressor heads
#         classification_output = torch.sigmoid(self.classifier_head(features)).squeeze()
#         regression_output = self.regressor_head(features)
#         yaw_output = self.yaw_head(features)
#         # print('out', regression_output)
#         # Concatenate regression outputs and yaw
#         combined_output = torch.cat((regression_output[:, :2], yaw_output, regression_output[:, 2:]), dim=1)
        
#         return classification_output, combined_output
def train(model, dataloader, num_epochs):
    # preprocess = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize to 224x224 directly
    #  transforms.Grayscale(num_output_channels=3), 
    # transforms.ToTensor(),  # Convert to PyTorch Tensor
    # ])
    def custom_loss(classification_output, regression_output, target, classification_weight=1, regression_weight=1):
        # # Mask for entries with a star (not nan)
        # # Mask for entries with a star (not nan in the target)
        # star_present_target = ~torch.isnan(target[:, 0])
    
        # # Mask for valid predictions (not nan in the output)
        # star_present_output = ~torch.isnan(output[:, 0])
        # # Combined mask for valid entries in both output and target
        # valid_entries = star_present_target & star_present_output
        # if not valid_entries.any():
        #     return torch.tensor(0.0, device=output.device, requires_grad=True)
        # # Smooth L1 loss for regression - only applied where star is present
        # reg_loss = nn.SmoothL1Loss()
        # regression_loss = reg_loss(output[valid_entries], target[valid_entries])
        # # print('regression_loss ', regression_loss )

        # # Combined loss (in this case, only regression loss)
        # Binary cross-entropy loss for the classification head
        bce_loss = nn.BCEWithLogitsLoss()

        # Presence of a star is the inverse of the 'no star' probability
        star_present = ~(torch.isnan(target[:, 0]))
        classification_loss = bce_loss(classification_output.squeeze(), star_present.float())

        # Regression loss - only applied where star is present
        mse_loss = nn.MSELoss()

        # Mask for valid regression targets (not nan)
        valid_regression = star_present.unsqueeze(1).expand_as(regression_output)
        regression_loss = mse_loss(regression_output[valid_regression], target[valid_regression])

        # Combine losses with appropriate weighting
        total_loss = classification_weight * classification_loss + regression_weight * regression_loss
        return total_loss
    def eval(*, n_examples: int = 1024, current_model = None, preprocess =None) -> None:
        model = current_model
        model.eval()
        scores = []
        for _ in tqdm(range(n_examples)):
            image, label = synthesize_data()
        
            if preprocess: 
                image = Image.fromarray((image * 255).astype('uint8'), mode='L')
                image = preprocess(image)  # Preprocess the image
                image = image.to(DEVICE).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                classification_output, regression_output  = model(torch.Tensor(image[None,None]).to(DEVICE))
            # Apply a threshold to determine star presence
            threshold = 0.5  # Adjust as needed
            star_present = classification_output > threshold
            print('classification_output ', classification_output )
            # Prepare final output
            final_output = regression_output.clone()  # Make a copy to modify
            final_output[~star_present] = float('nan')  # Set to nan where star is not present
            print('final output', final_output)
            print('label', label)
            np_pred = final_output[0].detach().cpu().numpy()
            # print('prediction', np_pred )
            scores.append(score_iou(np_pred, label))
        ious = np.asarray(scores, dtype="float")
        ious = ious[~np.isnan(ious)]  # remove true negatives
        print((ious > 0.7).mean())
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = CustomLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(num_epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        model.train()
        for image, label in tqdm(dataloader, total=len(dataloader)):
            image = image.to(DEVICE).float()
            label = label.to(DEVICE).float()
            # print('image', image)
            optimizer.zero_grad()
            classification_output, regression_output = model(image)
            # print('classification_output ', classification_output )
            # print('regression_output ', regression_output )
            loss = custom_loss(classification_output, regression_output, label)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
        eval(current_model=model)
        print('loss is', np.mean(losses))
        # scheduler.step()
    return model


def main():
   
    # Replace the classifier with your own classifier and regressor
    # Let's assume the feature vector size output by MobileNet is 1280

    model = StarModel().to(DEVICE)
    # def preprocess_image(image):
    #     # Apply median blur
    #     # Assuming 'image' is your input floating-point image with pixel values in [0, 1]
    #     if image.dtype != np.uint8:
    #         # Convert the image to 8-bit (255 range) format if it's not already
    #         image_8bit = np.uint8(image * 255)
    #     else:
    #         image_8bit = image
    #     denoised_image = cv2.medianBlur(image_8bit, 5)
        
    #     # # Estimate the noise standard deviation from the image
    #     # sigma_est = np.mean(estimate_sigma(image_8bit))
    #     # # Apply non-local means denoising
    #     # patch_kw = dict(patch_size=5,      # 5x5 patches
    #     #                 patch_distance=6)
    #     # denoised_image = denoise_nl_means(denoised_image, h=1.15 * sigma_est, fast_mode=False, **patch_kw)
        
    #     # # Convert to binary if needed for line removal, adjust the threshold as needed
    #     # _, binary_image = cv2.threshold(denoised_image, 30, 255, cv2.THRESH_BINARY)
    #     # # ... (apply Hough Transform and remove lines)
        
    #     # # Apply CLAHE
    #     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     # enhanced_image = clahe.apply(denoised_image.astype('uint8'))
        
    #     return  denoised_image
    # preprocessed_images = []
    # labels = []
    # num_images  = 50000
    # for i in range(num_images):
    #     if i%100 == 0:
    #         print('current image is', i)
    #     # Synthesize new image and label
    #     image, label = synthesize_data()
    #     # Preprocess the image
    #     # processed_image = preprocess_image(image)
        
    #     preprocessed_images.append(image)
    #     labels.append(label)
    # # Convert the lists of images and labels to tensors
    # tensor_images = torch.stack([torch.Tensor(image) for image in preprocessed_images])
    # tensor_labels = torch.stack([torch.Tensor(label) for label in labels])

    # # Create a TensorDataset
    # dataset = TensorDataset(tensor_images.unsqueeze(1), tensor_labels)  # Add channel dimension
    # preprocess = transforms.Compose([
    # transforms.Resize((224, 224)),  # Resize to 224x224 directly
    #  transforms.Grayscale(num_output_channels=3), 
    #   # Convert to PyTorch Tensor
    # ])
    # Create a DataLoader
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    star_model = train(
        model,
        dataloader = torch.utils.data.DataLoader(StarDataset(), batch_size=64, num_workers=8),
        num_epochs=30,
    )
    torch.save(star_model.state_dict(), "model.pickle")


if __name__ == "__main__":
    main()
