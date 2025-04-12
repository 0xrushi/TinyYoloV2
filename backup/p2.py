import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torch import nn

# VOC dataset classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(Dataset):
    def __init__(self, root_dir, year="2012", image_set="train", img_size=416, transform=None):
        """
        PASCAL VOC Dataset for YOLO
        
        Args:
            root_dir (str): Root directory of the VOC dataset (should contain VOCdevkit)
            year (str): Year of the dataset ('2007' or '2012')
            image_set (str): 'train', 'val', or 'test'
            img_size (int): Size to resize images to
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.year = year
        self.image_set = image_set
        self.img_size = img_size
        self.transform = transform
        
        # VOC dataset directory structure
        self.image_dir = os.path.join(root_dir, f"VOCdevkit/VOC{year}/JPEGImages")
        self.annotation_dir = os.path.join(root_dir, f"VOCdevkit/VOC{year}/Annotations")
        
        # Get list of image ids from the dataset
        list_file = os.path.join(
            root_dir, f"VOCdevkit/VOC{year}/ImageSets/Main/{image_set}.txt"
        )
        with open(list_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
        
        # Set up grid parameters for YOLO
        self.S = self.img_size // 32  # Feature map size from TinyYOLOv2
        self.num_anchors = 5  # Number of anchor boxes
        self.num_classes = 20  # VOC classes
        
        # Anchors from the Tiny YOLO V2 model (scaled to 0-1)
        self.anchors = [
            (1.08/self.img_size, 1.19/self.img_size),
            (3.42/self.img_size, 4.41/self.img_size),
            (6.63/self.img_size, 11.38/self.img_size),
            (9.42/self.img_size, 5.11/self.img_size),
            (16.62/self.img_size, 10.52/self.img_size)
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        
        # Load image and annotations
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        boxes, labels = self._get_annotation(annotation_path)
        
        # Get original image dimensions
        height, width, _ = image.shape
        
        if self.transform:
            # Apply Albumentations transformations
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["class_labels"]
            
            # Convert boxes and labels to PyTorch tensors
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                
            if len(labels) > 0:
                labels = torch.tensor(labels)
            else:
                labels = torch.zeros(0)
        else:
            # Basic resizing and conversion when no transforms are applied
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if len(boxes) > 0:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                boxes[:, [0, 2]] *= self.img_size / width
                boxes[:, [1, 3]] *= self.img_size / height
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            
            labels = torch.tensor(labels) if labels else torch.zeros(0)
        
        # Encode boxes to YOLO format
        yolo_targets = self._encoder(boxes, labels)
        
        return image, yolo_targets

    def _get_annotation(self, annotation_path):
        """Parse VOC XML annotation file."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0
            class_name = obj.find("name").text.strip()
            
            # Skip difficult instances if you want
            # if difficult == 1:
            #     continue
                
            if class_name in self.class_to_idx:
                bbox = obj.find("bndbox")
                # VOC dataset format is [xmin, ymin, xmax, ymax]
                box = [
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text),
                ]
                boxes.append(box)
                labels.append(self.class_to_idx[class_name])
        
        return boxes, labels

    def _encoder(self, boxes, labels):
        """
        Encode bounding boxes to YOLO format.
        Output shape: [S, S, 5*B + C] where:
          - S: grid size
          - B: number of anchor boxes (5 in Tiny YOLO v2)
          - C: number of classes (20 for VOC)
        """
        S = self.S
        B = self.num_anchors
        C = self.num_classes
        
        # Initialize target tensor
        target = torch.zeros(S, S, B, 5 + C)
        
        if len(boxes) == 0:
            return target
        
        # Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height]
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0  # [x_center, y_center]
        box_sizes = boxes[:, 2:4] - boxes[:, 0:2]  # [width, height]
        
        # Find which cell each box belongs to
        cell_x = torch.floor(box_centers[:, 0] * S)
        cell_y = torch.floor(box_centers[:, 1] * S)
        
        # Convert box coordinates to be relative to the cell
        x = box_centers[:, 0] * S - cell_x
        y = box_centers[:, 1] * S - cell_y
        
        # Normalize box sizes by image size
        w = box_sizes[:, 0]
        h = box_sizes[:, 1]
        
        # For each ground truth, find best matching anchor
        for idx in range(len(boxes)):
            if int(cell_x[idx]) >= S or int(cell_y[idx]) >= S:
                continue  # Skip if box center falls outside grid
                
            c = int(cell_x[idx])
            r = int(cell_y[idx])
            
            # One-hot encoding for class label
            target[r, c, 5*B + int(labels[idx])] = 1
            
            box_w, box_h = w[idx], h[idx]
            
            # Find the best anchor box for this ground truth
            best_anchor_idx = self._get_best_anchor(box_w.item(), box_h.item())
            
            # Set objectness and box coordinates for the best anchor
            box_offset = best_anchor_idx * 5
            target[r, c, box_offset] = 1  # objectness
            target[r, c, box_offset + 1] = x[idx]
            target[r, c, box_offset + 2] = y[idx]
            target[r, c, box_offset + 3] = torch.log(w[idx] + 1e-16)  # avoid log(0)
            target[r, c, box_offset + 4] = torch.log(h[idx] + 1e-16)
            
        return target

    def _get_best_anchor(self, width, height):
        """Find the best anchor box for a given box dimensions."""
        width = float(width)
        height = float(height)
        best_iou = 0
        best_idx = 0
        
        for i, (anchor_w, anchor_h) in enumerate(self.anchors):
            # Calculate IoU between box and anchor
            inter_w = min(width, anchor_w)
            inter_h = min(height, anchor_h)
            inter_area = inter_w * inter_h
            
            box_area = width * height
            anchor_area = anchor_w * anchor_h
            union_area = box_area + anchor_area - inter_area
            
            iou = inter_area / union_area
            if iou > best_iou:
                best_iou = iou
                best_idx = i
                
        return best_idx


def get_data_loaders(data_root, img_size=416, batch_size=16, num_workers=4):
    """Create train and validation data loaders."""
    
    # Define transformations with albumentations
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    val_transforms = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Create datasets
    train_dataset = VOCDataset(
        root_dir=data_root,
        year="2012", 
        image_set='train',
        img_size=img_size,
        transform=train_transforms
    )
    
    val_dataset = VOCDataset(
        root_dir=data_root,
        year="2012", 
        image_set='val',
        img_size=img_size,
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, S=13, B=5, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        """
        YOLO Loss function
        
        Args:
            S (int): Grid size (e.g. 13x13 for TinyYOLOv2 with 416x416 input)
            B (int): Number of bounding boxes per grid cell
            C (int): Number of classes
            lambda_coord (float): Weight for coordinate loss
            lambda_noobj (float): Weight for no-object loss
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Calculate YOLO loss. Both predictions and targets are assumed to be in the shape:
            (batch_size, S, S, B, 5+C)
        where for each anchor:
            - index 0 is objectness
            - indices 1:5 are bbox coordinates (x, y, w, h)
            - indices 5: are class probabilities (one-hot encoded target)

        Args:
            predictions: (batch_size, S, S, B*5 + C) tensor, reshaped to (batch,S,S,B,5+C)
            targets: (batch_size, S, S, B, 5+C) tensor
        """
        batch_size = predictions.size(0)
        # Reshape predictions to have extra anchor dimension
        predictions = predictions.view(batch_size, self.S, self.S, self.B, 5 + self.C)
        # Here we assume targets already have shape (batch, S, S, B, 5+C)
        
        # Object mask: where an object is present (assumes target encoding sets this appropriately)
        obj_mask = targets[..., 0:1]    # shape: (batch, S, S, B, 1)
        noobj_mask = 1 - obj_mask        # shape: (batch, S, S, B, 1)
        
        # Coordinate loss: computed only where objects exist.
        coord_loss = self.lambda_coord * self.mse(
            obj_mask * predictions[..., 1:5],
            obj_mask * targets[..., 1:5]
        )
        
        # Confidence loss (objectness)
        conf_loss_obj = self.mse(
            obj_mask * predictions[..., 0:1],
            obj_mask * targets[..., 0:1]
        )
        conf_loss_noobj = self.lambda_noobj * self.mse(
            noobj_mask * predictions[..., 0:1],
            noobj_mask * targets[..., 0:1]
        )
        
        # Class loss: calculated only for cells where objects exist.
        class_loss = self.mse(
            obj_mask * predictions[..., 5:],
            obj_mask * targets[..., 5:]
        )
        
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
        
        return total_loss / batch_size

    def calculate_iou(self, pred_boxes, target_boxes):
        """
        Calculate IoU between pred_boxes and target_boxes
        Both should be in the format (x_center, y_center, w, h).
        """
        # Convert to corner coordinates [x1, y1, x2, y2]
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # Calculate intersection coordinates
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Intersection area
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + 1e-6
        
        return inter_area / union_area



# Training function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, 
                weight_decay=1e-5, checkpoint_dir='checkpoints'):
    """
    Train the TinyYOLOv2 model
    
    Args:
        model: TinyYOLOv2 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        checkpoint_dir: Directory to save checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create loss function and optimizer
    S = model.width // 32  # Grid size based on model width
    criterion = YOLOLoss(S=S, B=5, C=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"tiny_yolo_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"tiny_yolo_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training complete!")
    return model


# Helper function to visualize predictions
def visualize_predictions(model, dataloader, device, num_images=5):
    """Visualize model predictions on sample images."""
    model.eval()
    images_shown = 0
    
    for images, targets in dataloader:
        images = images.to(device)
        
        with torch.no_grad():
            predictions = model(images)
        
        # Convert predictions to bounding boxes
        # This is a simplified visualization - you'd need a proper decoding function
        for idx in range(images.size(0)):
            if images_shown >= num_images:
                return
                
            # Convert image back to numpy
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            
            # Visualize the image
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title("Sample with ground truth boxes")
            plt.axis('off')
            plt.show()
            
            images_shown += 1


# Main training script
def main():
    # Initialize model
    from model import TinyYoloV2  # Import your model
    
    # Set parameters
    img_size = 416
    batch_size = 16
    num_epochs = 50
    data_root = "data"  # Update with your actual path
    
    # Initialize model with correct dimensions
    model = TinyYoloV2()
    model.width = img_size
    model.height = img_size
    
    # Load pretrained weights if available
    try:
        model.load_weights("yolov2-tiny-voc.weights")
        print("Loaded pretrained weights")
    except:
        print("No pretrained weights found, starting from scratch")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        checkpoint_dir='checkpoints'
    )
    
    # Visualize some predictions
    visualize_predictions(model, val_loader, device, num_images=5)


if __name__ == "__main__":
    main()