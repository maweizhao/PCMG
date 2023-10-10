import torch
from tqdm import *

def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        #print(motion_loader.len())
        if hasattr(motion_loader,'len'):
            batch_num=motion_loader.len()
        else:
            batch_num=len(motion_loader)
        for batch in tqdm(motion_loader,total=batch_num):
            # print(batch["lengths"])
            batch_prob = classifier(batch["output_xyz"], lengths=batch["lengths"])
            
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1

    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion
