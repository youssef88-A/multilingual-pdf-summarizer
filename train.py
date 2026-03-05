"""
Fine-tuning script for multilingual summarization.
Command: python train.py --model_name csebuetnlp/mT5_multilingual_XLSum --batch_size 8 --lr 3e-5 --epochs 3
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import argparse
import logging
import os
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=1024, max_target_length=300):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Prepare inputs
        model_inputs = self.tokenizer(
            f"summarize: {item['text']}",
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                item['summary'],
                max_length=self.max_target_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        }

def train(args):
    # Set seeds
    set_seed(args.seed)
    
    # Initialize tensorboard
    writer = SummaryWriter(f'runs/{args.model_name.replace("/", "_")}')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load dataset (example with XSum dataset)
    logger.info("Loading dataset...")
    dataset = load_dataset('xsum')
    
    # Prepare datasets
    train_dataset = SummaryDataset(dataset['train'], tokenizer)
    val_dataset = SummaryDataset(dataset['validation'], tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + train_steps
            writer.add_scalar('Loss/train', loss.item(), global_step)
        
        avg_train_loss = train_loss / train_steps
        logger.info(f'Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        logger.info(f'Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}')
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = f'models/best_model_epoch_{epoch+1}.pt'
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, model_save_path)
            logger.info(f'Best model saved to {model_save_path}')
            
            # Log validation loss
            print(f"Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f} | Checkpoint: {model_save_path}")
    
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='csebuetnlp/mT5_multilingual_XLSum')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train(args)