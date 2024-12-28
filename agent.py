import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import firebase_admin
from firebase_admin import credentials, firestore
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer  # Switched to Hugging Face's tokenizer

# ----------------------------- #
#       Configuration Setup      #
# ----------------------------- #

# Initialize Firebase
def initialize_firebase():
    print("Initializing Firebase...")
    cred = credentials.Certificate("./key.json")  # Ensure this path is correct
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
        print("Firebase initialized.")
    else:
        print("Firebase already initialized.")
    db = firestore.client()
    return db

db = initialize_firebase()

# Initialize Spotify API
print("Initializing Spotify API...")
spotify_client_id = 'YOUR_SPOTIFY_CLIENT_ID'         # Replace with your Spotify Client ID
spotify_client_secret = 'YOUR_SPOTIFY_CLIENT_SECRET' # Replace with your Spotify Client Secret

def initialize_spotify():
    print("Setting up Spotify credentials...")
    credentials_manager = SpotifyClientCredentials(client_id=spotify_client_id, client_secret=spotify_client_secret)
    sp = spotipy.Spotify(auth_manager=credentials_manager)
    print("Spotify API initialized.")
    return sp

sp = initialize_spotify()

# ----------------------------- #
#          Data Classes          #
# ----------------------------- #

class RecommendationRequest(BaseModel):
    description: str  # Description provided by the user

# ----------------------------- #
#       Transformer Model        #
# ----------------------------- #

class MultiAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, context_size):
        super(MultiAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, context_size),
            nn.ReLU(),
            nn.Linear(context_size, embedding_dim)
        )
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Forward pass of the attention block

        Args:
            x (torch.Tensor): input tensor of shape (sequence_length, batch_size, embedding_dim)
        """
        attn_output, _ = self.attention(x, x, x)
        x = self.norm_1(attn_output + x)
        ff_output = self.feed_forward(x)
        x = self.norm_2(ff_output + x)
        return x

class MusicRecommendationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, context_size, num_layers):
        super(MusicRecommendationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([
            MultiAttentionBlock(embedding_dim, num_heads, context_size) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        """
        Forward pass of the Music Recommendation Model

        Args:
            x (torch.Tensor): input tensor containing token indices
        """
        x = self.embedding(x)  # Shape: (sequence_length, batch_size, embedding_dim)
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)  # Shape: (sequence_length, batch_size, vocab_size)
        return logits

# ----------------------------- #
#          Data Processing       #
# ----------------------------- #

def fetch_data(file_path='songs.json'):
    """
    Loads songs from a JSON file.

    Args:
        file_path (str): Path to the songs.json file.

    Returns:
        pd.DataFrame: DataFrame containing songs data.
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    songs_df = pd.DataFrame(songs)
    # Rename 'id' to 'song_id' for consistency
    songs_df.rename(columns={'id': 'song_id'}, inplace=True)
    print("Data loaded successfully.")
    return songs_df

def clean_data(songs_df):
    """
    Cleans and preprocesses the songs DataFrame.

    Args:
        songs_df (pd.DataFrame): Raw songs DataFrame.

    Returns:
        pd.DataFrame: Cleaned songs DataFrame.
    """
    print("Cleaning data...")
    songs_df.dropna(subset=['song_name', 'artist_name'], inplace=True)
    songs_df['genre'] = songs_df['genre'].str.lower()
    songs_df['sub_genre'] = songs_df['sub_genre'].str.lower().fillna('')
    songs_df['mood'] = songs_df['mood'].str.lower().fillna('')
    # Check if 'state' exists; if not, use 'country' or appropriate field
    if 'state' in songs_df.columns:
        songs_df['location'] = songs_df['city'].fillna('') + ', ' + songs_df['state'].fillna('')
    elif 'country' in songs_df.columns:
        songs_df['location'] = songs_df['city'].fillna('') + ', ' + songs_df['country'].fillna('')
    else:
        songs_df['location'] = songs_df['city'].fillna('')
    print("Data cleaned.")
    return songs_df

def preprocess_data(songs):
    """
    Preprocesses the songs data into a format suitable for training.

    Args:
        songs (pd.DataFrame): DataFrame of song dictionaries.

    Returns:
        pd.DataFrame: Processed DataFrame.
        dict: Song to index mapping.
    """
    print("Preprocessing data...")
    # Ensure we work on a copy to avoid SettingWithCopyWarning
    songs_df = songs[['song_id', 'song_name', 'artist_name', 'genre', 'sub_genre', 'mood', 'location']].copy()
    
    # Create a textual representation for each song
    def create_textual_data(row):
        return f"Song '{row['song_name']}' by {row['artist_name']} is a {row['genre']} song with a {row['mood']} mood. Sub-genre: {row['sub_genre']}. Located in {row['location']}. Recommend a similar song:"
    
    print("Creating textual data for each song.")
    songs_df.loc[:, 'text'] = songs_df.apply(create_textual_data, axis=1)
    
    # Assign unique indices for songs
    song_ids = songs_df['song_id'].unique().tolist()
    song_to_idx = {song_id: idx for idx, song_id in enumerate(song_ids)}
    print("Mapping song IDs to indices.")
    songs_df.loc[:, 'song_idx'] = songs_df['song_id'].map(song_to_idx)
    
    print("Data preprocessing complete.")
    return songs_df, song_to_idx

# ----------------------------- #
#        Dataset Class           #
# ----------------------------- #

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        print("Initializing MusicDataset...")
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
            'labels': input_ids.clone()  # For language modeling, labels are the same as input_ids
        }

# ----------------------------- #
#          Model Training        #
# ----------------------------- #

def train_model(model, dataloader, criterion, optimizer, device, scheduler=None, epochs=3):
    """
    Trains the model.

    Args:
        model (nn.Module): The recommendation model.
        dataloader (DataLoader): DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run the training on.
        scheduler: Learning rate scheduler.
        epochs (int): Number of training epochs.
    """
    print("Starting model training...")
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids.transpose(0, 1))  # Transformer expects (seq, batch, embed)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    # Save the trained model
    torch.save(model.state_dict(), "trained_music_recommender.pt")
    print("Model training complete and saved.")

# ----------------------------- #
#          Recommendation Logic  #
# ----------------------------- #

def recommend_songs_by_description(description, model, tokenizer, device, songs_df):
    """
    Generates a song recommendation based on user description.

    Args:
        description (str): User-provided description of desired song.
        model (nn.Module): Trained recommendation model.
        tokenizer: Tokenizer used for encoding.
        device: Device to run inference on.
        songs_df (pd.DataFrame): DataFrame containing song details.

    Returns:
        dict: Recommended song details if found, else fallback message.
    """
    print(f"Generating recommendation for description: '{description}'")
    input_text = description
    encoding = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids.transpose(0,1))
        predictions = outputs.logits
        predicted_id = torch.argmax(predictions, -1)
    
    recommended_text = tokenizer.decode(predicted_id.squeeze(), skip_special_tokens=True)
    print(f"Model generated text: '{recommended_text}'")
    # Extract the recommended song name from the generated text
    # This is a placeholder logic; you may need a better extraction method
    try:
        recommended_song_name = recommended_text.split("Recommend a similar song:")[-1].strip()
        print(f"Extracted recommended song name: '{recommended_song_name}'")
    except Exception as e:
        print(f"Error extracting song name: {e}")
        recommended_song_name = ""
    
    if recommended_song_name:
        recommended_song = songs_df[songs_df['song_name'].str.lower() == recommended_song_name.lower()]
        if not recommended_song.empty:
            print(f"Found recommended song: '{recommended_song.iloc[0]['song_name']}' by {recommended_song.iloc[0]['artist_name']}")
            return recommended_song.iloc[0].to_dict()
        else:
            print("Recommended song not found in dataset.")
    
    # Fallback: Return a random song if no match found
    random_song = songs_df.sample(1).iloc[0].to_dict()
    print(f"Fallback recommendation: '{random_song['song_name']}' by {random_song['artist_name']}")
    return {"song_name": random_song['song_name'], "artist_name": random_song['artist_name']}

def store_recommendation(recommendation):
    """
    Stores the recommended song in the 'ai' collection in Firebase.

    Args:
        recommendation (dict): Recommended song details.
    """
    print("Storing recommendation in Firebase...")
    recommendation_entry = {
        'song_id': recommendation.get('song_id', ''),
        'song_name': recommendation.get('song_name', ''),
        'artist_name': recommendation.get('artist_name', ''),
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    db.collection('ai').add(recommendation_entry)
    print("Recommendation stored in Firebase.")

# ----------------------------- #
#           API Setup            #
# ----------------------------- #

app = FastAPI()

@app.post("/get-recommendation/")
def get_recommendation(request: RecommendationRequest):
    try:
        description = request.description
        print(f"Received recommendation request with description: '{description}'")
        
        # Generate recommendation
        recommendation = recommend_songs_by_description(description, model, tokenizer, device, songs_df)
        
        if recommendation and recommendation.get('song_name'):
            # Optionally store the recommendation
            # store_recommendation(recommendation)
            print(f"Returning recommendation: '{recommendation['song_name']}' by {recommendation['artist_name']}'")
            return {"recommended_song": f"{recommendation['song_name']} by {recommendation['artist_name']}"}
        else:
            print("No similar songs found.")
            return {"recommended_song": "No similar songs found. Please try different criteria."}
    
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------- #
#          Run Server            #
# ----------------------------- #

if __name__ == "__main__":
    try:
        print("Starting the agent script...")
        # Load and preprocess data
        print("Loading songs data...")
        songs_df = fetch_data('songs.json')  # Ensure songs.json is in the correct path
        print("Cleaning songs data...")
        songs_df = clean_data(songs_df)
        
        # Preprocess songs
        print("Preprocessing songs data...")
        processed_df, song_to_idx = preprocess_data(songs_df)
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Switched to Hugging Face's tokenizer
        tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token, use eos_token as pad
        print("Tokenizer initialized.")
        
        # Prepare dataset
        print("Preparing dataset...")
        dataset = MusicDataset(processed_df['text'].tolist(), tokenizer, max_length=128)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        print("Dataset and DataLoader prepared.")
        
        # Initialize model
        print("Initializing model...")
        vocab_size = tokenizer.vocab_size  # Correct attribute for Hugging Face's tokenizer
        print(f"Vocabulary size: {vocab_size}")
        model = MusicRecommendationModel(vocab_size=vocab_size, embedding_dim=256, num_heads=8, context_size=512, num_layers=2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Model initialized and moved to {device}.")
        
        # Define loss and optimizer
        print("Defining loss function and optimizer...")
        unk_token = "<UNK>"
        try:
            unk_token_id = tokenizer.encode(unk_token)[0]
            print(f"Using UNK token ID: {unk_token_id}")
        except Exception as e:
            print(f"UNK token '{unk_token}' not found: {e}. Using -100 as ignore_index.")
            unk_token_id = -100
        criterion = nn.CrossEntropyLoss(ignore_index=unk_token_id)
        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        print("Loss function and optimizer defined.")
        
        # Train the model
        print("Commencing training...")
        train_model(model, dataloader, criterion, optimizer, device, scheduler, epochs=5)
        
        # Save the trained model
        print("Saving the trained model...")
        torch.save(model.state_dict(), "trained_music_recommender.pt")
        print("Model saved as 'trained_music_recommender.pt'.")
        
        # Load the trained model for inference
        print("Loading the trained model for inference...")
        model_load = MusicRecommendationModel(vocab_size=vocab_size, embedding_dim=256, num_heads=8, context_size=512, num_layers=2)
        model_load.load_state_dict(torch.load("trained_music_recommender.pt", map_location=device))
        model_load.to(device)
        model_load.eval()
        print("Model loaded and set to evaluation mode.")
        
        # Assign the loaded model to the global model variable used in the API
        model = model_load
        print("Assigned loaded model to global 'model' variable.")
        
        # Run the FastAPI server
        print("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"An error occurred while running the script: {e}")