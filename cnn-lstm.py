import os
import pandas as pd
from ast import literal_eval
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        # Load the CSV file into a DataFrame, specifying the separator and skipping the specified row
        self.df = pd.read_csv(csv_file, sep=';', skiprows=[0], converters={'labeled_train_SVO_ts': literal_eval})
        self.img_dir = img_dir
        # print(self.df.columns)
        print(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vid_name = self.df.iloc[idx, 0]
        labels = self.df.iloc[idx, 1]
        labels = eval(labels)
        
        if not labels:  # Check if labels list is empty
            labels_tensor = torch.tensor([], dtype=torch.float)
            final_label = torch.zeros(43, dtype=torch.float)
        else:
            labels_tensor = torch.Tensor(labels)
            final_label = torch.nn.functional.one_hot(labels_tensor.to(torch.int64), num_classes=43)
            final_label = final_label.sum(dim=0)
            final_label = final_label.clamp(max=1)

        #  path to the folder containing images for the video
        img_folder_path = os.path.join(self.img_dir, vid_name)
        
        # Load all images from the specified folder
        images = load_images_from_folder(img_folder_path)
        
        return images, final_label
    

def collate_fn(batch):
    images_list, labels_list = zip(*batch)
    max_length = max(len(img_seq) for img_seq in images_list)
    # Assuming images are RGB, 224x224
    images = torch.zeros(len(images_list), max_length, 3, 224, 224)

    for i, img_seq in enumerate(images_list):
        # Stack the image sequence and assign it to the correct position
        images[i, :len(img_seq)] = torch.stack(img_seq)

    labels = torch.stack(labels_list)
    return images, labels

        
    
def load_images_from_folder(folder):
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to 224x224 pixels
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #sorting the images in folder 
    filenames = sorted(os.listdir(folder))
    for filename in filenames:
        if filename.endswith(".jpg"):  
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                img_tensor = transform(img)
                images.append(img_tensor)
    # print(f"Type of images list: {type(images)}, Length: {len(images)}")
    return images



file_path = '/Users/Bi/Documents/Sproj/Updated_VidLife.csv'
img_dir = '/Users/Bi/Documents/Sproj/tvqa_data'

dataset = CustomDataset(csv_file=file_path, img_dir=img_dir)

from torch.utils.data import random_split

# Calculate the sizes of the splits
total_size = len(dataset)
train_size = int(0.7 * total_size)  # For example, 80% of the data for training
validation_size = int(0.15 * total_size)  # The remaining for validation
test_size = total_size - (train_size + validation_size) 

# Split the dataset
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# Creating data loaders for each set
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

item=dataset.__getitem__(0)



class CNN_RNN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(CNN_RNN, self).__init__()
        # Define your CNN layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define your RNN layers
        self.rnn = nn.LSTM(input_size=32 * 56 * 56, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 43)  # Adjust num_classes as necessary

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        # Initialize the hidden state for LSTM
        hidden = (torch.randn(num_layers, batch_size, hidden_dim).to(x.device),
                  torch.randn(num_layers, batch_size, hidden_dim).to(x.device))

        for i in range(seq_len):
            # Process each image in the sequence through CNN
            x_t = self.pool(F.relu(self.conv1(x[:, i])))
            x_t = self.pool(F.relu(self.conv2(x_t)))
            x_t = x_t.view(batch_size, -1)  # Flatten the features

            # Process the flattened image through LSTM
            out, hidden = self.rnn(x_t.unsqueeze(1), hidden)

        # Get features from the last LSTM output
        r_out = hidden[0][-1]

        # Pass through the fully connected layer
        out = self.fc(r_out)
        out = torch.sigmoid(out)
        return out
   



# Initialize the neural network
model_path = '/Users/Bi/Documents/Sproj/model_1.pth'
hidden_dim = 128
num_layers = 1
model = CNN_RNN(hidden_dim, num_layers)
model.load_state_dict(torch.load(model_path))
model.eval()
# model.train()

# model.load_state_dict(torch.load('/Users/Bi/Documents/Sproj/model.pth'))
# model.eval()

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the data loader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)



threshold = 0.086
correct = 0
total = 0
num_labels = 43
all_labels = []
all_predictions = []

true_positives = torch.zeros(num_labels)
false_negatives = torch.zeros(num_labels)
false_positives = torch.zeros(num_labels)
true_negatives = torch.zeros(num_labels)


model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        print("Predicted probabilities for one batch:", outputs)
        predicted = (outputs > threshold).float()  # Apply threshold to get binary predictions
        print("Predicted for one batch:", predicted)
        # print(predicted)
        print("Labels for one batch:", labels)
        # print(labels)
        correct += (predicted == labels).float().sum()  # Compare predictions with true labels
        total += labels.numel()  # Total number of label comparisons
        for i in range(num_labels):
            true_positives[i] += ((predicted[:, i] == 1) & (labels[:, i] == 1)).float().sum()
            false_positives[i] += ((predicted[:, i] == 1) & (labels[:, i] == 0)).float().sum()
            false_negatives[i] += ((predicted[:, i] == 0) & (labels[:, i] == 1)).float().sum()
            true_negatives[i] += ((predicted[:, i] == 0) & (labels[:, i] == 0)).float().sum()


# Calculate accuracy
accuracy = correct / total
recall_per_label = true_positives / (true_positives + false_negatives)
recall_per_label[torch.isnan(recall_per_label)] = 0
average_recall = torch.mean(recall_per_label).item()

precision_per_label = true_positives / (true_positives + false_positives)
precision_per_label[torch.isnan(precision_per_label)] = 0
average_precision = torch.mean(precision_per_label).item()
accuracy_per_label = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)




f1_score_per_label = 2 * (precision_per_label * recall_per_label) / (precision_per_label + recall_per_label)
f1_score_per_label[torch.isnan(f1_score_per_label)] = 0
average_f1_score = torch.mean(f1_score_per_label).item()

print('Accuracy: {:.2f}%'.format(100 * accuracy))
print('Average Recall: {:.2f}'.format(average_recall))
print('Average Precision: {:.2f}'.format(average_precision))
print('Average F1 Score: {:.2f}'.format(average_f1_score))

for i in range(num_labels):
    print(f'Recall for label {i}: {recall_per_label[i]:.2f}')
    print(f'Precision for label {i}: {precision_per_label[i]:.2f}')
    print(f'F1 Score for label {i}: {f1_score_per_label[i]:.2f}')
    print(f'Accuracy for label {i}: {accuracy_per_label[i].item():.2f}')
    

    

label_with_highest_f1 = torch.argmax(f1_score_per_label).item()
highest_f1_score = f1_score_per_label[label_with_highest_f1].item()

sorted_labels = torch.argsort(f1_score_per_label, descending=True)
sorted_indices = torch.argsort(precision_per_label, descending=True)

top_5_labels = sorted_labels[:5]
top_10_labels = sorted_labels[:10]
top_10_label_indices = sorted_indices[:10]

top_5_precision = torch.mean(precision_per_label[top_5_labels]).item()
top_5_recall = torch.mean(recall_per_label[top_5_labels]).item()
top_5_f1_score = torch.mean(f1_score_per_label[top_5_labels]).item()
average_accuracy_top_5 = torch.mean(accuracy_per_label[top_5_labels]).item()

top_10_precision = torch.mean(precision_per_label[top_10_labels]).item()
top_10_recall = torch.mean(recall_per_label[top_10_labels]).item()
top_10_f1_score = torch.mean(f1_score_per_label[top_10_labels]).item()
average_accuracy_top_10 = torch.mean(accuracy_per_label[top_10_labels]).item()

top_10_precision_scores = precision_per_label[top_10_label_indices]

for label, score in zip(top_10_label_indices, top_10_precision_scores):
    print(f'Label {label.item()}: Precision Score = {score.item():.2f}')


    
print(f'Label with the highest F1 score: {label_with_highest_f1} (F1 Score: {highest_f1_score:.2f})')
print(f'Average Precision for top 5 labels: {top_5_precision:.2f}')
print(f'Average Recall for top 5 labels: {top_5_recall:.2f}')
print(f'Average F1 Score for top 5 labels: {top_5_f1_score:.2f}')
print(f'Average Accuracy for Top 5 Labels: {average_accuracy_top_5:.2f}')

print("Top 20 Labels with Highest F1 Scores:")
for label in top_5_labels:
    print(f'Label {label.item()}: F1 Score = {f1_score_per_label[label].item():.2f}')
    

print(f'Average Precision for top 10 labels: {top_10_precision:.2f}')
print(f'Average Recall for top 10 labels: {top_10_recall:.2f}')
print(f'Average F1 Score for top 10 labels: {top_10_f1_score:.2f}')
print(f'Average Accuracy for Top 10 Labels: {average_accuracy_top_10:.2f}')






