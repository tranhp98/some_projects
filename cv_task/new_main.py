import numpy as np
import torch
from tqdm import tqdm
from train import StarResNet20, ResNetBlock
from utils import DEVICE, score_iou, synthesize_data
from PIL import Image
from torchvision import transforms
def load_model():
    model =StarResNet20(ResNetBlock, [3, 3, 3]).to(DEVICE)
    model.to(DEVICE)
    with open("model.pickle", "rb") as f:
        state_dict = torch.load(f, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# def eval(*, n_examples: int = 1024) -> None:
#     model = load_model()
#     scores = []
#     # preprocess = transforms.Compose([
#     # transforms.Resize((224, 224)),  # Resize to 224x224 directly
#     #  transforms.Grayscale(num_output_channels=3), 
#     # transforms.ToTensor(),  # Convert to PyTorch Tensor
#     # ])
#     for _ in tqdm(range(n_examples)):
#         image, label = synthesize_data(has_star= True)
#         # image = Image.fromarray((image * 255).astype('uint8'), mode='L')
#         # image = preprocess(image)  # Preprocess the image
#         # image = image.to(DEVICE).unsqueeze(0)  # Add batch dimension
#         with torch.no_grad():
#             pred = model(torch.Tensor(image[None,None]).to(DEVICE))
#         np_pred = pred[0].detach().cpu().numpy()
#         print('pred', np_pred)
#         print('label', label)
#         scores.append(score_iou(np_pred, label))

#     ious = np.asarray(scores, dtype="float")
#     ious = ious[~np.isnan(ious)]  # remove true negatives
#     print((ious > 0.7).mean())

def eval(*, n_examples: int = 1024, current_model = None, preprocess =None) -> None:
        model = load_model()
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
if __name__ == "__main__":
    eval()
