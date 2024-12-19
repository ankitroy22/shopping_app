# Shopping App Recommendation System Using OpenAI CLIP Model 🌐🚀

## Overview 🔍
This project demonstrates an **AI-powered recommendation system** that allows users to search for products using both **text** and **images**. By leveraging **OpenAI's CLIP model**, the app provides accurate and personalized shopping recommendations based on user input.

## Features 🌟
- **🖼 Image Search**: Upload a product image to find visually similar items in the inventory.
- **🔒 Text Search**: Input a product description to retrieve relevant recommendations.
- **🎮 Top Recommendations**: Displays the top 5 most relevant results based on similarity scores.

## How It Works ⚙️
1. **Data Processing**: 
   - Product images and descriptions are processed using **OpenAI's CLIP (ViT-B/32)** to generate embeddings.
   - These embeddings are stored in an **SQLite database** for efficient retrieval.

2. **Search Mechanism**:
   - For **🖼 image search**, embeddings of the uploaded image are compared to stored embeddings using **cosine similarity**.
   - For **🔒 text search**, the input description is tokenized, and its embedding is compared to stored embeddings.

3. **Recommendations**:
   - The system ranks results based on similarity scores and displays the top 5 matches.

## Technology Stack 🛠️
- **Programming Language**: Python 📚
- **Model**: OpenAI CLIP (ViT-B/32) 🌐
- **Database**: SQLite 📊
- **Interface**: Streamlit 🔄

## Setup and Installation 📦
1. Clone the repository:
   ```bash
   git clone https://github.com/ankitroy22/shopping_app.git
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage 🙋
1. **🖼 Image Search**:
   - Upload an image of the product you are looking for.
   - Click on "Search by Image" to view recommendations.

2. **🔒 Text Search**:
   - Enter a product description in the text box.
   - Click on "Search by Text" to view recommendations.

## Project Structure 🌍
```
shopping-app-recommendation-system/
├── app.py             # Streamlit app implementation
├── requirements.txt   # Dependencies
├── database/          # SQLite database and related files
├── utils/             # Utility scripts for data preprocessing
└── README.md          # Project documentation
```

## Example 🗃
- **🖼 Image Search**:
  Upload an image, and the system will display similar products along with their similarity scores.

- **🔒 Text Search**:
  Input a description like *"red sneakers"* to find matching items from the inventory.

## Future Enhancements 🌐
- Add support for larger datasets using cloud storage solutions.
- Implement advanced ranking algorithms for improved recommendation accuracy.
- Introduce user authentication and personalized search history.

## Acknowledgments 🙏
- **OpenAI** for the CLIP model.
- The open-source community for their invaluable tools and resources.

## License ✅
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📧
For queries or feedback, feel free to reach out via [ankitroy8521@gmail.com](mailto:ankitroy8521@gmail.com).
