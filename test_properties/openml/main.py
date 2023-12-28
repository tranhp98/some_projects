import openml
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,  OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score, classification_report
from properties_checker import compute_L2_norm,compute_L1_norm,compute_inner_product, cosine_similarity, compute_linear_approx, compute_smoothness
import wandb
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser(description='small-tasks')
parser.add_argument('--name', default='experiment', type=str, help='experiment name')
parser.add_argument('--optim', default=None, type=str, help='experiment name')
parser.add_argument('--save', default=False, type=bool, help='save param')
parser.add_argument('--check_opt', default=False, type=bool, help='save param')
args = parser.parse_args()


# Function to load and preprocess the dataset
def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return X, y
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)  # No softmax here
        x = self.fc1(x)
        return x
def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return accuracy, report
# Load the dataset
X, y = load_dataset(42396)  # ID for the 'aloi' dataset

# Preprocessing
# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler(with_mean= False)
X_scaled = scaler.fit_transform(X)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.astype(np.float32))
print('got here')
y_train_tensor = torch.tensor(y_train) 
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test)

# Create DataLoaders

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)
print(len(train_loader))
# Determine the number of features and classes
input_size = X_train.shape[1]
# print(len(np.unique(y_train)))
num_classes = len(np.unique(y_train))

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,  num_classes=num_classes)
model = model.to(device)
wandb.init(project='test_convexity', config=args, name=args.name)
wandb.watch(model)
# Hyperparameters
learning_rate = 2
num_epochs = 20

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
if args.check_opt:
    print('in here')
    checkpoint = torch.load('connect4/sgd_20.pth.tar')
    prev_param = checkpoint['current_param']
    prev_loss = checkpoint['current_loss']
    prev_grad = checkpoint['current_grad']
else:
    prev_param =[torch.zeros_like(p) for p in model.parameters()]
    prev_grad = [torch.zeros_like(p) for p in model.parameters()]
    prev_loss = 0
# filename = "aloi/adam_0.pth.tar"
# torch.save({'state_dict':optimizer.state_dict(), 'model_dict': model.state_dict() }, filename)
# Training loop
print(args.check_opt)
for epoch in range(num_epochs):
    print('currently in epoch', epoch)
    iteration = 0
    current_param =[torch.zeros_like(p) for p in model.parameters()]
    current_grad = [torch.zeros_like(p) for p in model.parameters()]
    linear_approx = 0
    convexity_gap = 0
    num = 0
    denom = 0
    exp_avg_L = 0
    exp_avg_gap = 0
    L = 0
    iterator = enumerate(train_loader)
    prev_batch = next(iterator)
    for i, (features, labels) in iterator:
        iteration += 1
        optimizer.zero_grad()
        #compute \nabla f(w_t,x_{t-1})
        if not args.check_opt:
            prev_batch_features = prev_batch[1][0].to(device)
            prev_batch_labels = prev_batch[1][1].to(device)
            prev_batch_outputs = model(prev_batch_features) 
            # print(prev_batch_labels -1)
            # print(prev_batch_outputs)
            prev_batch_loss = criterion(prev_batch_outputs, prev_batch_labels) #f(w_t,x_{t-1})
            current_loss = prev_batch_loss.item() 
            prev_batch_loss.backward()
            i = 0
            with torch.no_grad():
                for p in model.parameters():
                    current_grad[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                    current_param[i].copy_(p) #w_t
                    i+=1
            
            optimizer.zero_grad()    
        # move to device
        features, labels = features.to(device), labels.to(device)
        # Forward pass
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        if args.check_opt:
            i = 0
            with torch.no_grad():
                for p in model.parameters():
                    current_grad[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                    current_param[i].copy_(p) #w_t
                    i+=1
            current_loss = loss.item()
        if i>0:
            # print('prev_loss', prev_loss)
            linear_approx = compute_linear_approx(current_param, current_grad, prev_param)
            # get the smoothness constant, small L means function is relatively smooth
            current_L = compute_smoothness(current_param, current_grad, prev_param, prev_grad)
            L = max(L,current_L)
            # L = max(L,compute_smoothness(model, current_param, current_grad))
            # this is another quantity that we want to check: linear_approx / loss_gap. The ratio is positive is good
            num+= linear_approx
            denom+= current_loss - prev_loss # f(w_t,x_{t-1}) - f(w_{t-1},x_{t-1})
            current_convexity_gap = current_loss - prev_loss - linear_approx 
            exp_avg_gap = 0.9*exp_avg_gap + (1-0.9)*current_convexity_gap
            # exp_avg_gap_2 = 0.9999*exp_avg_gap_2 + (1-0.9999)*current_convexity_gap
            exp_avg_L = 0.9*exp_avg_L+ (1-0.9)*current_L
            # exp_avg_L_2 = 0.9999*exp_avg_L_2+ (1-0.9999)*current_L
            convexity_gap+= current_convexity_gap
        
        # # Backward and optimize
        # if not False:
        #     print('something')
        # print(args.check_opt)
        if not args.check_opt:
            # print('in here')
            i = 0
            with torch.no_grad():
                for p in model.parameters():
                    prev_grad[i].copy_(p.grad) #hold \nabla f(w_{t-1},x_{t-1}) for next iteration
                    prev_param[i].copy_(p) # hold w_{t-1 } for next iteration
                    i+=1
            prev_loss = loss.item()
        optimizer.step()
         
        prev_batch = (i, (features, labels))
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')
    if not args.check_opt:
        prev_batch_image = prev_batch[1][0].to(device)
        prev_batch_target = prev_batch[1][1].to(device)
        prev_batch_outputs = model(prev_batch_image) 
        prev_batch_loss = criterion(prev_batch_outputs, prev_batch_target) #f(w_t,x_{t-1})
        current_loss = prev_batch_loss.item() 
        prev_batch_loss.backward()
        i = 0
        with torch.no_grad():
            for p in model.parameters():
                current_grad[i].copy_(p.grad) #\nabla f(w_t,x_{t-1})
                current_param[i].copy_(p) #w_t
                i+=1
        # zero grad to do the actual update
        optimizer.zero_grad()
    wandb.log(
        {
            "train_loss": current_loss,
            "convexity_gap": convexity_gap/iteration,
            "smoothness": L,
            "linear/loss_gap": num/denom,
            "numerator" : num,
            "denominator": denom,
            'exp_avg_L_.9': exp_avg_L,
            "exp_avg_gap_.9":  exp_avg_gap, 
            'prev_loss': prev_loss
        }
    )
    if args.save:
            filename = "connect4/sgd_" + str(epoch+1) + ".pth.tar"
            torch.save({'state_dict':optimizer.state_dict(),'prev_grad':prev_grad, 
                            'prev_param': prev_param, 'current_grad':current_grad, 'current_param': current_param
                            , 'prev_loss':prev_loss , 'current_loss': current_loss
                          , 'model_dict': model.state_dict() }, filename)

accuracy, report = evaluate_model(model, test_loader, device)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
