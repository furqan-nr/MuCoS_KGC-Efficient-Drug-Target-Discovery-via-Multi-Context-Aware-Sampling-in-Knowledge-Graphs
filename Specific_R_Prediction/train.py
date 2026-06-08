import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW

from dataset import KGDataset
from utils import save_evaluation_results, _rank_batch, hard_negative_hinge_loss
import torch.nn.functional as F


def evaluate_model(model, dataloader, device, relation_to_idx, test_triplets, tokenizer):
    model.eval()
    rankings = []
    labels = []
    
    with torch.no_grad():
        for inputs, batch_labels in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_labels = batch_labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            rankings.extend(_rank_batch(logits, batch_labels))
    
    rankings = np.array(rankings)

    mrr = np.mean(1.0 / rankings)
    hits_at_1 = np.mean(rankings <= 1)
    hits_at_3 = np.mean(rankings <= 3)
    hits_at_10 = np.mean(rankings <= 10)
    
    return {
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@10': hits_at_10
    }


def train_and_evaluate(model_name, tokenizer_class, model_class, save_path, num_epochs,
                       train_triplets, valid_triplets, test_triplets,
                       relation_to_idx, all_triplets, device):
    
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=len(relation_to_idx))
    model.to(device)
    
    # Datasets & DataLoaders
    train_dataset = KGDataset(train_triplets, tokenizer, relation_to_idx, all_triplets)
    valid_dataset = KGDataset(valid_triplets, tokenizer, relation_to_idx, all_triplets)
    test_dataset  = KGDataset(test_triplets, tokenizer, relation_to_idx, all_triplets)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0
    patience = 3
    best_checkpoint_path = os.path.join(save_path, "checkpoint_best.pth")
    
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(**inputs)
                logits = outputs.logits
                loss_ce = F.cross_entropy(logits, labels)

                # simple hinge-based hard-negative emphasis
                hloss = hard_negative_hinge_loss(logits, labels, top_k=1, margin=0.5, weight=1.0)

                loss = loss_ce + hloss

                if torch.isnan(loss).any():
                    print("Loss is NaN. Skipping this batch.")
                    continue

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            except Exception as e:
                print(f"Error during training epoch {epoch+1}: {e}")
                continue
        
        avg_train_loss = train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")

        print("Evaluating on validation set...")
        val_results = evaluate_model(model, valid_dataloader, device, relation_to_idx, valid_triplets, tokenizer)
        print("Validation Results:")
        print(val_results)

        if val_results['MRR'] > best_val_mrr:
            best_val_mrr = val_results['MRR']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, best_checkpoint_path)
            print(f"Best validation MRR improved to {best_val_mrr:.4f}; saved checkpoint_best.pth")
        else:
            epochs_without_improvement += 1
            print(f"No validation MRR improvement for {epochs_without_improvement} epoch(s) (patience={patience}).")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epoch(s) without validation improvement.")
            break
        
        # Save checkpoint
        checkpoint_file = os.path.join(save_path, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")
    
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])

    # Final evaluation
    print("\nStarting final evaluation on test set...")
    eval_results = evaluate_model(model, test_dataloader, device, relation_to_idx, test_triplets, tokenizer)
    
    print("\nEvaluation Results:")
    print(eval_results)
    
    # Save final model and results
    model_save_file = os.path.join(save_path, "model.pth")
    torch.save(model.state_dict(), model_save_file)
    print(f"Final model saved to: {model_save_file}")
    
    save_evaluation_results(eval_results, save_path)