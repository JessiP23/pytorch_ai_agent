import streamlit as st
import torch
from transformers import GPT2Tokenizer
import pandas as pd
import json
from fuzzywuzzy import process, fuzz

# Import your existing classes and functions
from agent import MusicRecommendationModel, InferenceModel, clean_text, make_clickable

# Load the model
@st.cache_resource
def load_model():
    model_path = "best_trained_music_recommender.pt"
    tokenizer_path = "gpt2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return InferenceModel(model_path=model_path, tokenizer_path=tokenizer_path, device=device)

# Load the songs dataset
@st.cache_data
def load_songs_data():
    songs_path = "songs.json"
    with open(songs_path, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    songs_df = pd.DataFrame(songs)
    songs_df['song_name_clean'] = songs_df['song_name'].apply(clean_text)
    songs_df['artist_name_clean'] = songs_df['artist_name'].apply(clean_text)
    return songs_df

# Streamlit app
def main():
    st.title("Music Recommendation System")

    # Load model and data
    model = load_model()
    songs_df = load_songs_data()

    # User input
    user_input = st.text_area("Describe the type of music you're looking for:", height=100)

    if st.button("Get Recommendation"):
        if user_input:
            with st.spinner("Generating recommendation..."):
                recommendation = model.generate_recommendation(user_input, songs_df)
            
            # Display recommendation
            st.subheader("Recommended Song")
            st.write(f"**Song:** {recommendation['song_name']}")
            st.write(f"**Artist:** {recommendation['artist_name']}")
            st.write(f"**Genre:** {recommendation['genre']}")
            st.write(f"**Mood:** {recommendation['mood']}")
            st.write(f"**Country:** {recommendation['country']}")
            
            # Display URLs as clickable links
            if recommendation['spotify_url']:
                st.markdown(f"[Listen on Spotify]({recommendation['spotify_url']})")
            if recommendation['apple_url']:
                st.markdown(f"[Listen on Apple Music]({recommendation['apple_url']})")
            if recommendation['instagram_url']:
                st.markdown(f"[Artist's Instagram]({recommendation['instagram_url']})")
        else:
            st.warning("Please enter a description to get a recommendation.")

if __name__ == "__main__":
    main()

