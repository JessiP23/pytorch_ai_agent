# standard libraries
# stream handling
import sys 
#JSON handling
import json
# regular expressions
import re
# math functions
import math
# asynchroneous context manager
import os
import asyncio

# PyTorch and transformers
# build and train neural networks
import torch
import torch.nn as nn
# language generation tasks
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# optimizer (weight decary and learning rate scheduling)    
from torch.optim import AdamW  
# improve convergence and reduce training time
from transformers import get_linear_schedule_with_warmup

# data handling
import pandas as pd
# split data into training and testing sets
from sklearn.model_selection import train_test_split

# Batching and loading data
from torch.utils.data import Dataset, DataLoader

# log warnings and errors
import logging

# API building for trained model
from fastapi import FastAPI, HTTPException
# Serialization and data validation
from pydantic import BaseModel
# ASGI server to run FastAPI app
import uvicorn
from typing import Optional

import warnings
from fuzzywuzzy import process, fuzz  # Added imports for fuzzy matching

# FastAPI lifespan management
from contextlib import asynccontextmanager 

# progress bars
from tqdm import tqdm

# ----------------------------- #
#       Configuration Setup      #
# ----------------------------- #

# Suppress specific FutureWarnings from PyTorch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Initialize Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensures logs are output to stdout in Colab
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------- #
#        Utility Functions       #
# ----------------------------- #

# normalize and clean text
def clean_text(text):
    """Normalize and clean text for consistency."""
    
    # handling cases where text is not a string
    if not isinstance(text, str):
        text = str(text)
        
    # Remove text within parentheses and brackets
    # Regex pattern: \(.*?\) OR \[.*?\]
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    return re.sub(r'[^\w\s]', '', text).strip().lower()


# calculate perplexity based on the loss value
def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    try:
        return math.exp(loss)
    
    # if the loss is too large, return infinity
    except OverflowError:
        logger.error("Overflow error encountered while calculating perplexity.")
        return float('inf')

def make_clickable(url):
    """Format URLs as clickable links."""
    # clikcable link
    return f"<{url}>" if url else ""

# ----------------------------- #
#          Dataset Class         #
# ----------------------------- #

# inheritances from Dataset class
class SongDataset(Dataset):
    
    # constructor method
    def __init__(self, descriptions, targets, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.targets = targets
        # convert text into token IDs
        self.tokenizer = tokenizer
        self.max_length = max_length


    # return the length of the dataset
    def __len__(self):
        return len(self.descriptions)

    # return the item at the given index
    def __getitem__(self, idx):
        # match idx to string
        description = str(self.descriptions[idx])
        target = str(self.targets[idx])

        # Encode description
        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            # return output as PyTorch tensors
            return_tensors='pt',
        )

        # Encode target
        # Prepare the model input
        target_encoding = self.tokenizer.encode_plus(
            target,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Changed from flatten() to squeeze()
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# ----------------------------- #
#          Model Class           #
# ----------------------------- #


# Neural Network Class
# GPT2LMHeadModel
class MusicRecommendationModel(nn.Module):
    # dropout to prevent overfitting
    def __init__(self, pretrained_model_name='gpt2', dropout=0.3):
        super(MusicRecommendationModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        # dropout layer of 0.3
        # fraction of the neuron output to 0 during training
        self.dropout = nn.Dropout(dropout)
        # Fixed the error: Removed len() since vocab_size is already an integer
        self.model.resize_token_embeddings(self.model.config.vocab_size)
    
    # compute the output of the model given the input
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # output.logits is the raw and unnormilized prediction values for each token
        return outputs.loss, outputs.logits

# ----------------------------- #
#          Training Function     #
# ----------------------------- #

# epoch: one complete pass through the dataset

# model: neural network to be trained
# data_loader: Pytorch DataLoader for the training dataset
# optimizer: update the model's parameters based on the loss
# scheduler: adjust the learning rate during training
# scaler: mixed-precision training with AMP
def train_epoch(model, data_loader, optimizer, scheduler, device, epoch, scaler):
    
    # dropout and batch normalization layers behave differently during training
    model.train()
    total_loss = 0
    
    # progress bar for the current epoch
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch}", leave=True)
    
    # Batch procesing
    for batch in progress_bar:
        # move input_ids, attention_mask, and labels to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Reset gradients to zero to prevent accumulation
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda'):
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        # clip gradients to a maximum norm of 1.0 to stabilizer training and prevent exploding the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    perplexity = calculate_perplexity(avg_loss)
    logger.info(f"Epoch {epoch}: Training Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

def eval_model(model, data_loader, device, epoch, phase="Validation"):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"{phase} Epoch {epoch}", leave=True)
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type='cuda'):
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    perplexity = calculate_perplexity(avg_loss)
    logger.info(f"Epoch {epoch}: {phase} Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

# ----------------------------- #
#            Inference Class     #
# ----------------------------- #

class InferenceModel:
    def __init__(self, model_path, tokenizer_path='gpt2', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from '{model_path}' to device '{self.device}'...")
        self.model = MusicRecommendationModel()
        
        # Use weights_only=True if you're using PyTorch 2.0+
        if hasattr(torch, 'load') and 'weights_only' in torch.load.__code__.co_varnames:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loading tokenizer from '{tokenizer_path}'...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Model and tokenizer loaded successfully.")
    
    def generate_recommendation(self, description, songs_df, threshold=80):
        """Generate a song recommendation based on the description."""
        try:
            # Clean and encode the input description
            description_clean = clean_text(description)
            inputs = self.tokenizer.encode_plus(
                description_clean,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=50,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )

            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model generated text: '{predicted_text}'")

            # Extract song name and artist
            parts = predicted_text.split(' by ')
            if len(parts) >= 2:
                song_name = clean_text(parts[0])
                artist_name = clean_text(parts[1])
                logger.info(f"Extracted Song: '{song_name}', Artist: '{artist_name}'")
            else:
                logger.warning("Incomplete extraction; proceeding to fallback recommendation.")
                return self.fallback_recommendation(songs_df)
    
            # Fuzzy matching for song and artist
            song_matches = process.extract(
                song_name,
                songs_df['song_name_clean'],
                scorer=fuzz.token_sort_ratio,
                limit=5
            )

            # Iterate over song matches and find artist matches
            for song, song_score in song_matches:
                potential_songs = songs_df[songs_df['song_name_clean'] == song]
                for _, row in potential_songs.iterrows():
                    artist_score = fuzz.token_sort_ratio(artist_name, row['artist_name_clean'])
                    if artist_score >= threshold:
                        logger.info(f"Matched Song: '{row['song_name']}' by '{row['artist_name']}' with scores {song_score}, {artist_score}")
                        return self.format_recommendation(row)

            logger.warning("No suitable match found; proceeding to fallback recommendation.")
            return self.fallback_recommendation(songs_df)

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return self.fallback_recommendation(songs_df)

    def fallback_recommendation(self, songs_df):
        """Provide a random song as a fallback recommendation."""
        random_song = songs_df.sample(1).iloc[0]
        logger.info(f"Fallback Recommendation: '{random_song['song_name']}' by '{random_song['artist_name']}'")
        return self.format_recommendation(random_song)

    def format_recommendation(self, song):
        """Format the song recommendation."""
        return {
            "song_name": song['song_name'],
            "artist_name": song['artist_name'],
            "genre": song.get('genre', ''),
            "sub_genre": song.get('sub_genre', ''),
            "mood": song.get('mood', ''),
            "country": song.get('country', ''),
            "city": song.get('city', ''),
            "spotify_url": make_clickable(song.get('spotify_url', '')),
            "apple_url": make_clickable(song.get('apple_url', '')),
            "instagram_url": make_clickable(song.get('instagram_url', '')),
            "perplexity": ""  # Placeholder; calculate if needed
        }

# ----------------------------- #
#            FastAPI Setup       #
# ----------------------------- #

app = FastAPI(title="Music Recommendation API")

class RecommendationRequest(BaseModel):
    description: str

class RecommendationResponse(BaseModel):
    description: str
    recommended_song: str
    artist: str
    genre: str
    sub_genre: str
    mood: str
    country: str
    city: str
    spotify_url: Optional[str] = None
    apple_url: Optional[str] = None
    instagram_url: Optional[str] = None
    perplexity: Optional[float] = None

# Initialize Inference Model
inference_model = None
songs_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_model, songs_df
    try:
        logger.info("Starting up FastAPI application...")

        # Paths to model and dataset
        model_path = "best_trained_music_recommender.pt"
        tokenizer_path = "gpt2"
        songs_path = "songs.json"

        # Load the tokenizer
        logger.info(f"Loading tokenizer from '{tokenizer_path}'...")
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully.")

        # Initialize and load the model
        logger.info(f"Loading model from '{model_path}'...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inference_model = InferenceModel(model_path=model_path, tokenizer_path=tokenizer_path, device=device)

        # Load the songs dataset
        logger.info(f"Loading songs data from '{songs_path}'...")
        with open(songs_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        songs_df = pd.DataFrame(songs)

        # Clean song and artist names for matching
        songs_df['song_name_clean'] = songs_df['song_name'].apply(clean_text)
        songs_df['artist_name_clean'] = songs_df['artist_name'].apply(clean_text)
        logger.info(f"Loaded {len(songs_df)} songs for inference.")

        logger.info("FastAPI application startup complete.")

        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down FastAPI server...")

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_song(request: RecommendationRequest):
    if not inference_model or songs_df is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    recommendation = inference_model.generate_recommendation(request.description, songs_df)
    return RecommendationResponse(
        description=request.description,
        recommended_song=recommendation["song_name"],
        artist=recommendation["artist_name"],
        genre=recommendation["genre"],
        sub_genre=recommendation["sub_genre"],
        mood=recommendation["mood"],
        country=recommendation["country"],
        city=recommendation["city"],
        spotify_url=recommendation["spotify_url"],
        apple_url=recommendation["apple_url"],
        instagram_url=recommendation["instagram_url"],
        perplexity=recommendation.get("perplexity")
    )


# ----------------------------- #
#            Main Function       #
# ----------------------------- #

def main():
    """Main function to train the music recommendation model."""
    try:
        logger.info("Starting the training script...")
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load the tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token
        logger.info("Tokenizer loaded.")
        
        # Load the songs dataset
        songs_path = "songs.json"  # Ensure that songs.json is in the same directory
        logger.info(f"Loading songs data from '{songs_path}'...")
        with open(songs_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        songs_df_local = pd.DataFrame(songs)
        
        # Data Cleaning and Preparation
        logger.info("Cleaning and preparing songs data...")
        songs_df_local['description'] = songs_df_local.apply(
            lambda row: f"{row['genre']} song by {row['artist_name']} from {row['country']}. Mood: {row.get('mood', 'neutral')}.",
            axis=1
        )
        songs_df_local['target'] = songs_df_local.apply(
            lambda row: f"{row['song_name']} by {row['artist_name']}",
            axis=1
        )
        
        # Normalize text
        songs_df_local['description'] = songs_df_local['description'].apply(clean_text)
        songs_df_local['target'] = songs_df_local['target'].apply(clean_text)
        
        # Handle missing values
        songs_df_local = songs_df_local.fillna({
            'mood': 'neutral',
            'genre': 'Unknown',
            'artist_name': 'Unknown',
            'song_name': 'Unknown'
        })
        
        # Split data into training, validation, and testing sets (70% train, 20% val, 10% test)
        logger.info("Splitting data into Train (70%), Validation (20%), and Test (10%) sets...")
        train_df, temp_df = train_test_split(songs_df_local, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)  # 0.3 * (1/3) = 0.1
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
        
        # Create datasets
        train_dataset = SongDataset(
            descriptions=train_df['description'].tolist(),
            targets=train_df['target'].tolist(),
            tokenizer=tokenizer,
            max_length=128
        )
        
        val_dataset = SongDataset(
            descriptions=val_df['description'].tolist(),
            targets=val_df['target'].tolist(),
            tokenizer=tokenizer,
            max_length=128
        )

        test_dataset = SongDataset(
            descriptions=test_df['description'].tolist(),
            targets=test_df['target'].tolist(),
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Create dataloaders
        batch_size = 8
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced num_workers to 2
            pin_memory=True
        )  # Reduced num_workers to 2
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize the model
        logger.info("Initializing model...")
        model = MusicRecommendationModel(pretrained_model_name='gpt2', dropout=0.3)
        model.to(device)
        logger.info("Model initialized.")
        
        # Define optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)  # Changed to PyTorch's AdamW and reduced learning rate
        epochs = 10
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Initialize mixed precision scaler
        scaler = torch.amp.GradScaler()  # Changed to torch.amp.GradScaler()
    
        # Training loop with validation and checkpointing
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs}")
            train_loss, train_perplexity = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler)
            val_loss, val_perplexity = eval_model(model, val_loader, device, epoch, phase="Validation")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(model.state_dict(), "best_trained_music_recommender.pt")
                logger.info(f"Validation loss improved. Model checkpoint saved.")
            else:
                patience_counter += 1
                logger.info(f"No improvement in validation loss for {patience_counter} epoch(s).")
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

        logger.info("Training completed.")

        # Evaluate on Test Set
        logger.info("Evaluating model on Test Set...")
        test_loss, test_perplexity = eval_model(model, test_loader, device, epoch, phase="Test")
        logger.info(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}")

        # Start the FastAPI server after training
        logger.info("Starting FastAPI server...")

        import threading

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info("FastAPI server is running in the background.")

    except Exception as e:
        logger.error(f"An error occurred in the training script: {e}")

# ----------------------------- #
#            Entry Point         #
# ----------------------------- #

if __name__ == "__main__":
    main()