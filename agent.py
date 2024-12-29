import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import math
import nest_asyncio  # To handle asyncio issues in certain environments

# ----------------------------- #
#       Configuration Setup      #
# ----------------------------- #

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------- #
#       Transformer Model        #
# ----------------------------- #

class MusicRecommendationModel(nn.Module):
    def __init__(self, pretrained_model_name='gpt2', dropout=0.3):
        super(MusicRecommendationModel, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        self.gpt2.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained(pretrained_model_name)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits

# ----------------------------- #
#        Data Processing         #
# ----------------------------- #

def fetch_data(file_path):
    logger.info(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    songs_df = pd.DataFrame(songs)
    if 'id' in songs_df.columns:
        songs_df.rename(columns={'id': 'song_id'}, inplace=True)
    logger.info("Data loaded successfully.")
    return songs_df

def clean_data(songs_df):
    logger.info("Cleaning data...")
    songs_df.dropna(subset=['song_name', 'artist_name'], inplace=True)
    songs_df['genre'] = songs_df['genre'].str.lower()
    songs_df['sub_genre'] = songs_df['sub_genre'].str.lower().fillna('')
    songs_df['mood'] = songs_df['mood'].str.lower().fillna('')
    if 'state' in songs_df.columns:
        songs_df['location'] = songs_df['city'].fillna('') + ', ' + songs_df['state'].fillna('')
    elif 'country' in songs_df.columns:
        songs_df['location'] = songs_df['city'].fillna('') + ', ' + songs_df['country'].fillna('')
    else:
        songs_df['location'] = songs_df['city'].fillna('')
    logger.info("Data cleaned.")
    return songs_df

def preprocess_data(songs_df):
    logger.info("Preprocessing data...")
    songs_df['text'] = songs_df.apply(
        lambda row: f"Song '{row['song_name']}' by {row['artist_name']}' is a {row['genre']} song with a {row['mood']} mood. Sub-genre: {row['sub_genre']}. Located in {row['location']}. Recommend a similar song:",
        axis=1
    )
    
    train_texts, val_texts = train_test_split(
        songs_df['text'].tolist(),
        test_size=0.1,
        random_state=42
    )
    logger.info("Data preprocessing complete.")
    return train_texts, val_texts

# ----------------------------- #
#        Dataset Class           #
# ----------------------------- #

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        logger.info("Initializing MusicDataset...")
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()  # shape: (max_length)
        attention_mask = encoding['attention_mask'].squeeze()  # shape: (max_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

# ----------------------------- #
#          Model Training        #
# ----------------------------- #

def calculate_perplexity(loss):
    return math.exp(loss)

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=10, patience=3):
    logger.info("Starting model training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        perplexity = calculate_perplexity(avg_val_loss)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_trained_music_recommender.pt")
            logger.info("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in validation loss for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break
    # Save the final model
    torch.save(model.state_dict(), "trained_music_recommender.pt")
    logger.info("Model training complete and saved.")

# ----------------------------- #
#          Recommendation Logic #
# ----------------------------- #

def generate_recommendation(description, model, tokenizer, device, songs_df):
    logger.info(f"Generating recommendation for: {description}")
    encoding = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        predicted_id = torch.argmax(logits, dim=-1)
    
    recommended_text = tokenizer.decode(predicted_id[0], skip_special_tokens=True)
    logger.info(f"Recommended Text: {recommended_text}")
    
    try:
        recommended_song_name = recommended_text.split("Recommend a similar song:")[-1].strip()
        logger.info(f"Extracted recommended song name: '{recommended_song_name}'")
    except Exception as e:
        logger.error(f"Error extracting song name: {e}")
        recommended_song_name = ""
    
    if recommended_song_name:
        recommended_song = songs_df[songs_df['song_name'].str.lower() == recommended_song_name.lower()]
        if not recommended_song.empty:
            logger.info(f"Found recommended song: '{recommended_song.iloc[0]['song_name']}' by {recommended_song.iloc[0]['artist_name']}'")
            return recommended_song.iloc[0].to_dict()
        else:
            logger.warning("Recommended song not found in dataset.")
    
    # Fallback: Return a random song if no match found
    random_song = songs_df.sample(1).iloc[0].to_dict()
    logger.info(f"Fallback recommendation: '{random_song['song_name']}' by {random_song['artist_name']}'")
    return {"song_name": random_song['song_name'], "artist_name": random_song['artist_name']}

def store_recommendation(db, recommendation):
    logger.info("Storing recommendation in Firebase...")
    recommendation_entry = {
        'song_id': recommendation.get('song_id', ''),
        'song_name': recommendation.get('song_name', ''),
        'artist_name': recommendation.get('artist_name', ''),
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    db.collection('ai').add(recommendation_entry)
    logger.info("Recommendation stored in Firebase.")

# ----------------------------- #
#           API Setup            #
# ----------------------------- #

app = FastAPI()
global model, tokenizer, device, songs_df, db

class RecommendationRequest(BaseModel):
    description: str  # Description provided by the user

@app.post("/get-recommendation/")
def get_recommendation(request: RecommendationRequest):
    try:
        description = request.description
        logger.info(f"Received recommendation request with description: '{description}'")
        
        # Generate recommendation
        recommendation = generate_recommendation(description, model, tokenizer, device, songs_df)
        
        if recommendation and recommendation.get('song_name'):
            # Optionally store the recommendation
            # store_recommendation(db, recommendation)
            logger.info(f"Returning recommendation: '{recommendation['song_name']}' by {recommendation['artist_name']}'")
            return {"recommended_song": f"{recommendation['song_name']} by {recommendation['artist_name']}"}
        else:
            logger.warning("No similar songs found.")
            return {"recommended_song": "No similar songs found. Please try different criteria."}
    
    except Exception as e:
        logger.error(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- #
#          Main Execution        #
# ----------------------------- #

def main():
    try:
        logger.info("Starting the agent script...")
        
        # Initialize Firebase
        logger.info("Initializing Firebase...")
        key_path = "./key.json"  # Update this path if necessary
        cred = credentials.Certificate(key_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized.")
        else:
            logger.info("Firebase already initialized.")
        db = firestore.client()
        
        # Load and preprocess data
        logger.info("Loading songs data...")
        songs_df = fetch_data('songs.json')  # Ensure songs.json is in the correct path
        logger.info("Cleaning songs data...")
        songs_df = clean_data(songs_df)
        
        logger.info("Preprocessing songs data...")
        train_texts, val_texts = preprocess_data(songs_df)
        
        # Initialize tokenizer
        logger.info("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token
        logger.info("Tokenizer initialized.")
        
        # Create Training and Validation Datasets
        logger.info("Creating Training Dataset...")
        train_dataset = MusicDataset(train_texts, tokenizer, max_length=128)
        
        logger.info("Creating Validation Dataset...")
        val_dataset = MusicDataset(val_texts, tokenizer, max_length=128)
        
        # Create DataLoaders
        logger.info("Creating DataLoaders...")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        logger.info("DataLoaders created.")
        
        # Initialize model
        logger.info("Initializing model...")
        model = MusicRecommendationModel(pretrained_model_name='gpt2', dropout=0.3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        logger.info(f"Model initialized and moved to {device}.")
        
        # Define optimizer and scheduler
        logger.info("Defining optimizer and scheduler...")
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        total_steps = len(train_dataloader) * 10  # epochs=10
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        logger.info("Optimizer and scheduler defined.")
        
        # Train the model
        logger.info("Commencing training...")
        train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=10, patience=3)
        
        # Save the trained model
        logger.info("Saving the trained model...")
        torch.save(model.state_dict(), "trained_music_recommender.pt")
        logger.info("Model saved as 'trained_music_recommender.pt'.")
        
        # Load the trained model for inference
        logger.info("Loading the trained model for inference...")
        model_load = MusicRecommendationModel(pretrained_model_name='gpt2', dropout=0.3)
        model_load.load_state_dict(torch.load("trained_music_recommender.pt", map_location=device))
        model_load.to(device)
        model_load.eval()
        logger.info("Model loaded and set to evaluation mode.")
        
        # Assign the loaded model to the global model variable used in the API
        model = model_load
        logger.info("Assigned loaded model to global 'model' variable.")
        
        # Apply nest_asyncio to handle asyncio issues in certain environments
        try:
            nest_asyncio.apply()
            logger.info("Applied nest_asyncio to allow nested event loops.")
        except Exception as e:
            logger.warning(f"Could not apply nest_asyncio: {e}")
        
        # Run the FastAPI server
        logger.info("Starting FastAPI server...")
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000)
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                logger.warning("RuntimeError encountered: asyncio.run() cannot be called from a running event loop. Applying nest_asyncio and retrying...")
                nest_asyncio.apply()
                uvicorn.run(app, host="0.0.0.0", port=8000)
            else:
                raise e
        
    except Exception as e:
        logger.error(f"An error occurred while running the script: {e}")

if __name__ == "__main__":
    main()