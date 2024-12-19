import json
import clip
import torch
import streamlit as st
from io import BytesIO
import sqlite3
import numpy as np
from PIL import Image as PILImage
from sklearn.metrics.pairwise import cosine_similarity
from tempfile import NamedTemporaryFile

# Load CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Function to preprocess input text
def preprocess_text(text, max_words=70):
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    text_tensor = clip.tokenize(text).to(device)
    return text_tensor


# Function to search similar images
def search_similar_images(image_path, top_k=5):
    # Connect to the SQLite database
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()

    try:
        image = preprocess(PILImage.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy().flatten()

        cursor.execute("SELECT data FROM image_data")
        data = cursor.fetchall()

        similarities = []
        for row in data:
            record = json.loads(row[0])
            combined_vector = np.array(record['combined_vector'])
            similarity = cosine_similarity([image_features], [combined_vector])
            record['similarity'] = similarity[0][0]
            similarities.append(record)

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:top_k]

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

    finally:
        conn.close()


# Streamlit app
st.title("Shopping App")

# Image search
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if st.button("Search by Image"):
        image = PILImage.open(uploaded_file)
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_file.name)  # Save temporarily
        results = search_similar_images(temp_file.name)

        if results:
            st.write("Recommended Images:")
            cols = st.columns(2)  # Display images in two columns

            for i, record in enumerate(results):
                with cols[i % 2]:
                    st.image(record['image'], caption=f"Similarity: {record['similarity']:.2f}", use_container_width=True)
                    st.write(record)
        else:
            st.write("No similar images found.")

# Text search
text_input = st.text_input("Enter text to search:")
if text_input:
    if st.button("Search by Text"):
        text_tensor = preprocess_text(text_input).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tensor)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().flatten()

        conn = sqlite3.connect('mydatabase.db')
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT data FROM image_data")
            data = cursor.fetchall()

            similarities = []
            for row in data:
                record = json.loads(row[0])
                combined_vector = np.array(record['combined_vector'])
                similarity = cosine_similarity([text_features], [combined_vector])
                record['similarity'] = similarity[0][0]
                similarities.append(record)

            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            st.write("Top 5 Results:")
            cols = st.columns(2)

            for i, record in enumerate(similarities[:5]):
                with cols[i % 2]:
                    st.image(record['image'], caption=f"Similarity: {record['similarity']:.2f}", use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            conn.close()
