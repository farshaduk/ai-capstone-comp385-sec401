"""
BERT-based Fraud Detection Model

This module implements a REAL AI fraud detection system using:
1. DistilBERT for text classification (fine-tuned on rental fraud data)
2. Proper train/validation/test splits
3. Real evaluation metrics (accuracy, precision, recall, F1)
4. Model checkpointing and versioning

This is the core AI component of the fraud detection system.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Transformers imports
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# Get the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BACKEND_DIR, "models", "bert_fraud_models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for BERT fine-tuning"""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42


@dataclass
class TrainingMetrics:
    """Metrics from model training and evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    training_loss_history: List[float]
    validation_loss_history: List[float]
    best_epoch: int
    total_training_time: float


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertFraudClassifier:
    """
    BERT-based Fraud Detection Classifier
    
    This is a REAL AI model that:
    1. Fine-tunes DistilBERT on labeled fraud data
    2. Learns semantic patterns of fraudulent listings
    3. Provides probability scores with real trained weights
    
    This is NOT keyword matching - it understands context and meaning.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to load a pre-trained model. If None, will use base model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = TrainingConfig()
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.model_path = model_path
        self.metrics = None
        
        logger.info(f"Using device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _initialize_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Loading {self.config.model_name}...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
        logger.info("Model initialized successfully")
    
    def load_dataset(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split dataset into train/val/test.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Loading dataset from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Ensure required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=df['label']
        )
        
        # Second split: train vs val
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.config.random_seed,
            stratify=train_val['label']
        )
        
        logger.info(f"Dataset split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        logger.info(f"Label distribution - Train: {train['label'].value_counts().to_dict()}")
        
        return train, val, test
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config: Optional[TrainingConfig] = None
    ) -> TrainingMetrics:
        """
        Fine-tune BERT on the fraud detection task.
        
        This is WHERE THE AI LEARNING HAPPENS.
        The model learns to distinguish fraud from legitimate listings
        based on semantic patterns, not just keywords.
        
        Args:
            train_df: Training data with 'text' and 'label' columns
            val_df: Validation data
            config: Training configuration
        
        Returns:
            TrainingMetrics with all evaluation results
        """
        import time
        start_time = time.time()
        
        if config:
            self.config = config
        
        # Initialize model if not already done
        if self.model is None:
            self._initialize_model()
        
        # Create datasets
        train_dataset = FraudDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        val_dataset = FraudDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        training_loss_history = []
        validation_loss_history = []
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        logger.info("=" * 60)
        logger.info("Starting BERT Fine-tuning for Fraud Detection")
        logger.info("=" * 60)
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                               f"Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            training_loss_history.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            validation_loss_history.append(avg_val_loss)
            
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Training Loss: {avg_train_loss:.4f}")
            logger.info(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict().copy()
                logger.info(f"  ✓ New best model!")
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time:.1f} seconds")
        logger.info(f"Best epoch: {best_epoch}")
        logger.info("=" * 60)
        
        # Create initial metrics (will be updated after test evaluation)
        self.metrics = TrainingMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            roc_auc=0.0,
            confusion_matrix=[],
            training_loss_history=training_loss_history,
            validation_loss_history=validation_loss_history,
            best_epoch=best_epoch,
            total_training_time=total_time
        )
        
        return self.metrics
    
    def evaluate(self, test_df: pd.DataFrame) -> TrainingMetrics:
        """
        Evaluate the trained model on test set.
        
        This provides REAL metrics on unseen data.
        
        Args:
            test_df: Test data with 'text' and 'label' columns
        
        Returns:
            TrainingMetrics with all evaluation results
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        logger.info("Evaluating model on test set...")
        
        test_dataset = FraudDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            self.config.max_length
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Fraud probability
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = 0.0
        
        cm = confusion_matrix(all_labels, all_preds).tolist()
        
        # Update metrics
        if self.metrics:
            self.metrics.accuracy = accuracy
            self.metrics.precision = precision
            self.metrics.recall = recall
            self.metrics.f1_score = f1
            self.metrics.roc_auc = roc_auc
            self.metrics.confusion_matrix = cm
        else:
            self.metrics = TrainingMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                confusion_matrix=cm,
                training_loss_history=[],
                validation_loss_history=[],
                best_epoch=0,
                total_training_time=0.0
            )
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"ROC AUC:   {roc_auc:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
        logger.info("=" * 60)
        
        return self.metrics
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict fraud probability for a single listing.
        
        This uses the TRAINED BERT model to understand the text
        semantically - not just keyword matching.
        
        Args:
            text: The listing text to analyze
        
        Returns:
            Dict with 'is_fraud', 'fraud_probability', 'confidence'
        """
        if self.model is None:
            # Try to load default model
            default_path = os.path.join(MODELS_DIR, "latest")
            if os.path.exists(default_path):
                self.load_model(default_path)
            else:
                raise ValueError("No trained model available. Train a model first.")
        
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            fraud_prob = probs[0][1].item()
            is_fraud = fraud_prob > 0.5
            confidence = max(fraud_prob, 1 - fraud_prob)
        
        return {
            'is_fraud': is_fraud,
            'fraud_probability': round(fraud_prob, 4),
            'legitimate_probability': round(1 - fraud_prob, 4),
            'confidence': round(confidence, 4),
            'prediction_label': 'fraud' if is_fraud else 'legitimate'
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict fraud probability for multiple listings"""
        return [self.predict(text) for text in texts]
    
    def save_model(self, path: str):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config and metrics
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        if self.metrics:
            metrics_path = os.path.join(path, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2)
        
        # Update latest symlink
        latest_path = os.path.join(MODELS_DIR, "latest")
        if os.path.exists(latest_path):
            if os.path.islink(latest_path):
                os.unlink(latest_path)
            else:
                import shutil
                shutil.rmtree(latest_path)
        
        # Create symlink or copy
        try:
            os.symlink(path, latest_path)
        except:
            import shutil
            shutil.copytree(path, latest_path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        logger.info(f"Loading model from {path}")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
        self.model = DistilBertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        
        # Load config
        config_path = os.path.join(path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = TrainingConfig(**config_dict)
        
        # Load metrics
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_dict = json.load(f)
                self.metrics = TrainingMetrics(**metrics_dict)
        
        self.is_trained = True
        self.model_path = path
        
        logger.info("Model loaded successfully")


def train_fraud_model(dataset_path: str = None, output_name: str = None) -> Dict[str, Any]:
    """
    Convenience function to train a new fraud detection model.
    
    Args:
        dataset_path: Path to CSV with 'text' and 'label' columns
        output_name: Name for the saved model
    
    Returns:
        Dict with training results and metrics
    """
    if dataset_path is None:
        dataset_path = os.path.join(BACKEND_DIR, "data", "fraud_dataset.csv")
    
    if output_name is None:
        output_name = f"bert_fraud_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_path = os.path.join(MODELS_DIR, output_name)
    
    # Initialize classifier
    classifier = BertFraudClassifier()
    
    # Load and split data
    train_df, val_df, test_df = classifier.load_dataset(dataset_path)
    
    # Train
    classifier.train(train_df, val_df)
    
    # Evaluate on test set
    metrics = classifier.evaluate(test_df)
    
    # Save model
    classifier.save_model(output_path)
    
    return {
        'model_path': output_path,
        'metrics': asdict(metrics),
        'dataset_size': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
    }


# Singleton instance for easy import
_classifier_instance = None

def get_fraud_classifier() -> BertFraudClassifier:
    """Get or create the global fraud classifier instance"""
    global _classifier_instance
    
    if _classifier_instance is None:
        # Try to load the latest trained model
        latest_path = os.path.join(MODELS_DIR, "latest")
        if os.path.exists(latest_path):
            _classifier_instance = BertFraudClassifier(latest_path)
        else:
            _classifier_instance = BertFraudClassifier()
    
    return _classifier_instance
