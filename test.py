import json
import torch
from transformers import GPT2Tokenizer
import pandas as pd
import logging
import math
from fuzzywuzzy import process, fuzz
import re
from agent import MusicRecommendationModel  


# perplexity
# Measures how well a probability model predicts a sample (next word in a sentence)
# lower perplexity means the model is better at predicting the sample

# Lower perplexity
# The model is more confident
# The model is more accurate

# Higher perplexity
# The model is less confident
# The model is less accurate
# Lacks understanding of the data


"""
High Perplexity
1. Insufficient training
    - Not enough epochs
2. Learning Rate Issues
    - Too high learning rate
    - Optimal parameters
3. Data Quality
    - Noisy data
    - Inconsistent data
4. Model Complexity
    - Model is too simple
    - Model is too complex
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------- #
#        Utility Functions       #
# ----------------------------- #

def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        logger.error("Overflow error encountered while calculating perplexity.")
        return float('inf')

def clean_text(text):
    """Normalize and clean text for consistency."""
    return re.sub(r'[^\w\s]', '', text).strip().lower()

# ----------------------------- #
#          Recommendation Logic #
# ----------------------------- #

def generate_recommendation(description, model, tokenizer, device, songs_df, threshold=80):
    """
    Generate song recommendation based on user description.
    
    Args:
        description (str): User-provided description.
        model (nn.Module): Trained music recommendation model.
        tokenizer (Tokenizer): GPT-2 tokenizer.
        device (torch.device): Computation device.
        songs_df (DataFrame): DataFrame containing song data.
        threshold (int): Minimum match score for fuzzy matching.
        
    Returns:
        dict: Recommended song data with mood, genre, etc.
    """
    logger.info(f"Generating recommendation for: '{description}'")
    
    # Encode the input description
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
    
    # Generate model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss, logits = outputs
        perplexity = calculate_perplexity(loss.item())
        predicted_id = torch.argmax(logits, dim=-1)
    
    # Decode the predicted tokens to text
    recommended_text = tokenizer.decode(predicted_id[0], skip_special_tokens=True)
    logger.info(f"Recommended Text: '{recommended_text}'")
    
    # Extract song name and artist using ' by ' separator
    parts = recommended_text.split(' by ')
    if len(parts) >= 2:
        recommended_song_name = clean_text(parts[0])
        recommended_artist_name = clean_text(parts[1])
        logger.info(f"Extracted recommended song name: '{recommended_song_name}' and artist name: '{recommended_artist_name}'")
    else:
        recommended_song_name = ""
        recommended_artist_name = ""
        logger.warning("Could not extract recommended song name and artist name from the model's output.")
    
    # Initialize recommendation dictionary
    recommendation = {
        "song_name": "",
        "artist_name": "",
        "genre": "",
        "sub_genre": "",
        "mood": "",
        "country": "",
        "city": "",
        "spotify_url": "",
        "apple_url": "",
        "instagram_url": "",
        "perplexity": perplexity
    }
    
    if recommended_song_name:
        # Fuzzy matching to find the best song match
        song_names = songs_df['song_name_clean'].tolist()
        best_match, score = process.extractOne(
            recommended_song_name, 
            song_names, 
            scorer=fuzz.token_sort_ratio
        )
        
        if score >= threshold:
            # Retrieve the song data
            recommended_song = songs_df[songs_df['song_name_clean'] == best_match].iloc[0]
            recommendation.update({
                "song_name": recommended_song['song_name'],
                "artist_name": recommended_song['artist_name'],
                "genre": recommended_song.get('genre', ''),
                "sub_genre": recommended_song.get('sub_genre', ''),
                "mood": recommended_song.get('mood', ''),
                "country": recommended_song.get('country', ''),
                "city": recommended_song.get('city', ''),
                "spotify_url": recommended_song.get('spotify_url', ''),
                "apple_url": recommended_song.get('apple_url', ''),
                "instagram_url": recommended_song.get('instagram_url', ''),
                "perplexity": perplexity
            })
            logger.info(f"Found recommended song: '{recommended_song['song_name']}' by '{recommended_song['artist_name']}' with score {score}")
        else:
            logger.warning(f"No suitable match found for '{recommended_song_name}'. Highest score: {score}")
    else:
        logger.warning("No song name extracted; proceeding to fallback recommendation.")
    
    # Fallback: Return a random song if no match found
    if recommendation["song_name"] == "":
        random_song = songs_df.sample(1).iloc[0]
        recommendation.update({
            "song_name": random_song['song_name'],
            "artist_name": random_song['artist_name'],
            "genre": random_song.get('genre', ''),
            "sub_genre": random_song.get('sub_genre', ''),
            "mood": random_song.get('mood', ''),
            "country": random_song.get('country', ''),
            "city": random_song.get('city', ''),
            "spotify_url": random_song.get('spotify_url', ''),
            "apple_url": random_song.get('apple_url', ''),
            "instagram_url": random_song.get('instagram_url', ''),
            "perplexity": perplexity
        })
        logger.info(f"Fallback recommendation: '{random_song['song_name']}' by '{random_song['artist_name']}'")
    
    return recommendation

# ----------------------------- #
#            Main Function       #
# ----------------------------- #

def main():
    """Main function to test the recommendation system."""
    try:
        logger.info("Starting the test script...")
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load the tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # GPT2 does not have a pad token
        logger.info("Tokenizer loaded.")
        
        # Initialize the model
        logger.info("Initializing model...")
        model = MusicRecommendationModel(pretrained_model_name='gpt2', dropout=0.3)
        model.to(device)
        logger.info("Model initialized.")
        
        # Load the trained model weights
        model_path = "best_trained_music_recommender.pt"  # Use the best model
        logger.info(f"Loading model weights from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info("Model loaded and set to evaluation mode.")
        
        # Load the songs dataset
        songs_path = "songs.json"  # Ensure that songs.json is in the same directory
        logger.info(f"Loading songs data from '{songs_path}'...")
        with open(songs_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        songs_df = pd.DataFrame(songs)
        
        # Data Cleaning and Preparation
        logger.info("Cleaning and preparing songs data...")
        songs_df['song_name_clean'] = songs_df['song_name'].apply(clean_text)
        songs_df['artist_name_clean'] = songs_df['artist_name'].apply(clean_text)
        
        # Drop rows with missing essential information
        songs_df.dropna(subset=['song_name_clean', 'artist_name_clean'], inplace=True)
        
        logger.info(f"Total songs loaded: {len(songs_df)}")
        
        # Example descriptions to generate recommendations
        test_descriptions = [
            "I enjoy upbeat pop songs with a strong rhythm.",
            "I'm looking for relaxing acoustic tracks.",
            "Give me some energetic rock music."
        ]
        
        for desc in test_descriptions:
            recommendation = generate_recommendation(desc, model, tokenizer, device, songs_df)
            print(f"Description: {desc}")
            if recommendation["song_name"]:
                print(f"Recommended Song: {recommendation['song_name']} by {recommendation['artist_name']}")
                print(f"Genre: {recommendation['genre']}")
                print(f"Sub-Genre: {recommendation['sub_genre']}")
                print(f"Mood: {recommendation['mood']}")
                print(f"Country: {recommendation['country']}")
                print(f"City: {recommendation['city']}")
                print(f"Spotify: {recommendation['spotify_url']}")
                print(f"Apple: {recommendation['apple_url']}")
                print(f"Instagram: {recommendation['instagram_url']}")
                print(f"Perplexity: {recommendation['perplexity']:.2f}\n")
            else:
                print("No recommendation could be made at this time.\n")
    
    except Exception as e:
        logger.error(f"An error occurred in the test script: {e}")

if __name__ == "__main__":
    main()