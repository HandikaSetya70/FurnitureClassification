import torch
import os
import random
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from helper_logger import DataLogger
from model_base import SimpleCNN
from helper_tester import ModelTesterMetrics
from dataset import SimpleTorchDataset
from torchvision import transforms

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
SEED = 424242
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

#torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_epochs = 100
batch_size = 64

if __name__ == "__main__":
    print("| Pytorch Model Training !")
    
    print("| Total Epoch :", total_epochs)
    print("| Batch Size  :", batch_size)
    print("| Device      :", device)

    logger = DataLogger("FurnitureClassification")
    metrics = ModelTesterMetrics()

    metrics.loss = torch.nn.BCEWithLogitsLoss()
    metrics.activation = torch.nn.Softmax(1)

    # Using SimpleCNN with 5 output classes (chair, cupboard, fridge, table, tv)
    model = SimpleCNN(7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Optional training augmentations
    training_augmentation = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip()
    ]

    # Initialize datasets with the new processed data paths
    validation_dataset = SimpleTorchDataset('./processed_dataset/val')
    training_dataset = SimpleTorchDataset('./processed_dataset/train', training_augmentation)
    testing_dataset = SimpleTorchDataset('./processed_dataset/test')

    # Initialize dataloaders
    validation_datasetloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    training_datasetloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    testing_datasetloader = DataLoader(testing_dataset, batch_size=1, shuffle=True)

    # Training Evaluation Loop
    for current_epoch in range(total_epochs):
        print("Epoch :", current_epoch)
        
        # Training Loop
        model.train()  # set the model to train mode
        metrics.reset()  # reset the metrics

        for (image, label) in tqdm(training_datasetloader, desc="Training :"):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = metrics.compute(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        training_mean_loss = metrics.average_loss()
        training_mean_accuracy = metrics.average_accuracy()

        # Evaluation Loop
        model.eval()  # set the model to evaluation mode
        metrics.reset()  # reset the metrics

        with torch.no_grad():  # disable gradient computation for evaluation
            for (image, label) in tqdm(validation_datasetloader, desc="Validation:"):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                metrics.compute(output, label)

        evaluation_mean_loss = metrics.average_loss()
        evaluation_mean_accuracy = metrics.average_accuracy()

        # Log the results
        logger.append(
            current_epoch,
            training_mean_loss,
            training_mean_accuracy,
            evaluation_mean_loss,
            evaluation_mean_accuracy
        )

        # Save the best model
        if logger.current_epoch_is_best:
            print("> Latest Best Epoch :", logger.best_accuracy())
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            state_dictionary = {
                "model_state": model_state,
                "optimizer_state": optimizer_state
            }
            torch.save(
                state_dictionary, 
                logger.get_filepath("best_checkpoint.pth")
            )

        logger.save()
        print("")
    
    print("| Training Complete, Loading Best Checkpoint")
    
    # Load Best Model State
    state_dictionary = torch.load(
        logger.get_filepath("best_checkpoint.pth"), 
        map_location=device
    )
    model.load_state_dict(state_dictionary['model_state'])
    model = model.to(device)
    
    # Final Testing Phase
    model.eval()  # set the model to evaluation mode
    metrics.reset()  # reset the metrics

    print("| Running Final Test Evaluation")
    
    with torch.no_grad():  # disable gradient computation for testing
        for (image, label) in tqdm(testing_datasetloader, desc="Testing:"):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            metrics.compute(output, label)

    testing_mean_loss = metrics.average_loss()
    testing_mean_accuracy = metrics.average_accuracy()

    # Log final results
    print("\n=== Final Results ===")
    logger.write_text(f"# Final Testing Loss     : {testing_mean_loss:.4f}")
    logger.write_text(f"# Final Testing Accuracy : {testing_mean_accuracy:.4f}")
    logger.write_text(f"# Classification Report:")
    logger.write_text(metrics.report())
    logger.write_text(f"# Confusion Matrix:")
    logger.write_text(metrics.confusion())
    print("\n")