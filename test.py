import sys
import json
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import math
import warnings
from fuzzywuzzy import process, fuzz
from tqdm import tqdm

# ----------------------------- #
#       Configuration Setup      #
# ----------------------------- #

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------- #
#        Utility Functions       #
# ----------------------------- #

def clean_text(text):
    """Normalize and clean text for consistency."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def calculate_perplexity(loss):
    """Calculate perplexity from loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        logger.error("Overflow error encountered while calculating perplexity.")
        return float('inf')

def make_clickable(url):
    """Format URLs as clickable links."""
    return f"<{url}>" if url else ""

# ----------------------------- #
#          Dataset Class         #
# ----------------------------- #

class SongDataset(Dataset):
    def __init__(self, descriptions, targets, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        target = str(self.targets[idx])

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

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
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

# ----------------------------- #
#          Model Class           #
# ----------------------------- #

class MusicRecommendationModel(nn.Module):
    def __init__(self, pretrained_model_name='gpt2', dropout=0.3):
        super(MusicRecommendationModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.model.resize_token_embeddings(self.model.config.vocab_size)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

# ----------------------------- #
#          Evaluation Function   #
# ----------------------------- #

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=True)
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).float().sum()
                total_correct += correct.item()
                total_samples += labels.numel()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{(total_correct/total_samples)*100:.2f}%"})
            except Exception as e:
                logger.error(f"Error during batch evaluation: {e}")
    
    avg_loss = total_loss / len(data_loader)
    perplexity = calculate_perplexity(avg_loss)
    accuracy = (total_correct / total_samples) * 100
    logger.info(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, perplexity, accuracy

# ----------------------------- #
#            Inference Class     #
# ----------------------------- #

class InferenceModel:
    def __init__(self, model_path, tokenizer_path='gpt2', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model from '{model_path}' to device '{self.device}'...")
        self.model = MusicRecommendationModel()
        try:
            model_state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(model_state_dict, dict) and 'model_state_dict' in model_state_dict:
                self.model.load_state_dict(model_state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded and set to evaluation mode.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

        logger.info(f"Loading tokenizer from '{tokenizer_path}'...")
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            sys.exit(1)
    
    def generate_recommendation(self, description, songs_df, threshold=90):
        """Generate a song recommendation based on the description."""
        try:
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

            # Adjust generation parameters
            outputs = self.model.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_beams=5,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model generated text: '{predicted_text}'")

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

            for match in song_matches:
                if len(match) >= 2:
                    song, song_score = match[:2]
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
            "perplexity": ""  # Can be calculated if needed
        }

# ----------------------------- #
#            Main Function       #
# ----------------------------- #

def main():
    """Main function to evaluate the music recommendation model."""
    try:
        logger.info("Starting the evaluation script...")
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load the tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded.")
        
        # Load the songs dataset
        songs_path = "songs.json"
        logger.info(f"Loading songs data from '{songs_path}'...")
        with open(songs_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        songs_df = pd.DataFrame(songs)
        
        # Data Cleaning and Preparation
        logger.info("Cleaning and preparing songs data...")
        songs_df['description'] = songs_df.apply(
            lambda row: f"{row['genre']} song by {row['artist_name']} from {row['country']}. Mood: {row.get('mood', 'neutral')}.",
            axis=1
        )
        songs_df['target'] = songs_df.apply(
            lambda row: f"{row['song_name']} by {row['artist_name']}",
            axis=1
        )
        
        # Normalize text
        songs_df['description'] = songs_df['description'].apply(clean_text)
        songs_df['target'] = songs_df['target'].apply(clean_text)
        
        # Handle missing values
        songs_df = songs_df.fillna({
            'mood': 'neutral',
            'genre': 'Unknown',
            'artist_name': 'Unknown',
            'song_name': 'Unknown'
        })
        
        # Split data into training, validation, and testing sets (70% train, 20% val, 10% test)
        logger.info("Splitting data into Train (70%), Validation (20%), and Test (10%) sets...")
        train_df, temp_df = train_test_split(songs_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)  # 0.3 * (1/3) = 0.1
        logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
        
        # Create datasets
        test_dataset = SongDataset(
            descriptions=test_df['description'].tolist(),
            targets=test_df['target'].tolist(),
            tokenizer=tokenizer,
            max_length=128
        )
        
        # Create dataloader
        batch_size = 8
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Initialize the model
        logger.info("Initializing model...")
        model = MusicRecommendationModel(pretrained_model_name='gpt2', dropout=0.3)
        model.to(device)
        logger.info("Model initialized.")
        
        # Load the trained model weights
        model_path = "best_trained_music_recommender.pt"  # Use the best model
        logger.info(f"Loading model weights from '{model_path}'...")
        try:
            model_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(model_state_dict)
            model.eval()
            logger.info("Model loaded and set to evaluation mode.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            sys.exit(1)
        
        # Evaluate the model on the test set
        logger.info("Starting evaluation on the test set...")
        test_loss, test_perplexity, test_accuracy = evaluate(model, test_loader, device)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.2f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Example of testing input recognition
        logger.info("Testing input recognition with sample descriptions...")
        sample_descriptions = [
            "I want a pop song from Canada",
            "Recommend a chill jazz track",
            "Looking for an upbeat electronic dance music",
            "Give me a classical piece from Germany",
            "Need a rock song with high energy"
        ]
        
        # Load songs dataframe for inference
        logger.info(f"Loading songs data from '{songs_path}' for inference...")
        with open(songs_path, 'r', encoding='utf-8') as f:
            songs = json.load(f)
        songs_df_inference = pd.DataFrame(songs)
        songs_df_inference['song_name_clean'] = songs_df_inference['song_name'].apply(clean_text)
        songs_df_inference['artist_name_clean'] = songs_df_inference['artist_name'].apply(clean_text)
        
        # Initialize Inference Model
        inference_model = InferenceModel(model_path, tokenizer_path='gpt2', device=device)
        
        for desc in sample_descriptions:
            recommendation = inference_model.generate_recommendation(desc, songs_df_inference)
            print(f"\nDescription: {desc}")
            print(f"Recommended Song: {recommendation['song_name']} by {recommendation['artist_name']}")
            print(f"Genre: {recommendation['genre']}")
            print(f"Sub-Genre: {recommendation['sub_genre']}")
            print(f"Mood: {recommendation['mood']}")
            print(f"Country: {recommendation['country']}")
            print(f"City: {recommendation['city']}")
            print(f"Spotify: {recommendation['spotify_url']}")
            print(f"Apple: {recommendation['apple_url']}")
            print(f"Instagram: {recommendation['instagram_url']}")
            print(f"Perplexity: {recommendation.get('perplexity', 'N/A')}")
        
        logger.info("Evaluation script completed successfully.")
    
    except Exception as e:
        logger.error(f"An error occurred in the evaluation script: {e}")

# ----------------------------- #
#            Entry Point         #
# ----------------------------- #

if __name__ == "__main__":
    main()