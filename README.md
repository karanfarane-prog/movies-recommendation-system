# movies-recommendation-system
🎬 Movie Recommendation System
A content-based movie recommendation engine built with Python, Streamlit, and scikit-learn. Find your next favorite movie based on genres, keywords, cast, director, and plot overview.

https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white
https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge

📌 Features
Feature	Description
🔍 Search by Movie	Enter a movie title and get 5 similar recommendations
🎭 Search by Genre	Browse movies from your favorite genre
🎯 Movie + Genre Combo	Narrow down recommendations by genre
⭐ User Ratings	View TMDB crowd scores and top-rated movies
🎬 Movie Posters	Visual recommendations powered by TMDB API
📊 Model Analysis	Understand how the recommendation engine works
🖱️ Clickable Cards	Click any movie card to view details on TMDB
🚀 Live Demo
Click here to try the app!
(Update this link after deployment)

📸 Screenshots
Home Page
https://screenshots/home.png

Recommendations
https://screenshots/recommendations.png

User Ratings
https://screenshots/ratings.png

Model Analysis
https://screenshots/model.png

🛠️ Tech Stack
Technology	Purpose
Python 3.8+	Backend logic
Streamlit	Web interface
pandas	Data manipulation
NumPy	Numerical operations
scikit-learn	TF-IDF Vectorization & Cosine Similarity
TMDB API	Movie posters & details
HTML/CSS	Custom styling
Pickle	Model serialization
📦 Installation
Prerequisites
Python 3.8 or higher

Git

TMDB API Key (free from TMDB)

Setup Instructions
Clone the repository

bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
Create a virtual environment

bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Add TMDB API Key
Create a file at .streamlit/secrets.toml:

toml
TMDB_API_KEY = "your_api_key_here"
Run the app

bash
streamlit run app.py
📊 How It Works
Recommendation Engine Architecture
The app uses content-based filtering to recommend movies based on similarity:

text
┌─────────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1.  DATA PREPROCESSING                                         │
│      ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│      │  TMDB Movies │────▶│  TMDB Credits│────▶│  Merge     │  │
│      └──────────────┘     └──────────────┘     └────────────┘  │
│                                                                 │
│  2.  FEATURE EXTRACTION                                         │
│      ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│      │   Genres     │     │   Keywords   │     │   Cast     │  │
│      └──────────────┘     └──────────────┘     └────────────┘  │
│      ┌──────────────┐     ┌──────────────┐                     │
│      │   Director   │     │   Overview   │                     │
│      └──────────────┘     └──────────────┘                     │
│                                                                 │
│  3.  TAG BUILDING                                               │
│      ┌─────────────────────────────────────────────────────┐    │
│      │  tags = overview + genres + keywords + cast + crew │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  4.  VECTORIZATION                                              │
│      ┌─────────────────────────────────────────────────────┐    │
│      │  TF-IDF Vectorizer (max_features=5000)             │    │
│      │  Converts tags into numeric vectors                │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  5.  SIMILARITY CALCULATION                                     │
│      ┌─────────────────────────────────────────────────────┐    │
│      │  Cosine Similarity Matrix (4806 x 4806)            │    │
│      │  Compares every movie against every other movie    │    │
│      └─────────────────────────────────────────────────────┘    │
│                                                                 │
│  6.  RECOMMENDATION                                             │
│      ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│      │  User Input  │────▶│  Find Index  │────▶│ Top 5 Most │  │
│      └──────────────┘     └──────────────┘     │  Similar   │  │
│                                                  └────────────┘  │
└─────────────────────────────────────────────────────────────────┘
Code Example
python
# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

# Extract features
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Build tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)
📁 Project Structure
text
movie-recommendation-system/
│
├── app.py                         # Main Streamlit application
├── recommend.py                   # Recommendation logic
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignored files
│
├── tmdb_5000_movies.csv           # Movie dataset (4803 movies)
├── tmdb_5000_credits.csv          # Credits dataset
│
├── module.ipynb                   # Jupyter notebook (development)
│
├── .streamlit/
│   └── secrets.toml               # API keys (DO NOT upload!)
│
└── README.md                      # Documentation
📊 Dataset Information
Dataset	Rows	Columns	Source
tmdb_5000_movies.csv	4,803	20	TMDB
tmdb_5000_credits.csv	4,803	4	TMDB
Features Used:
Genres: Movie genres (Action, Adventure, etc.)

Keywords: Important keywords related to the movie

Cast: Top 3 actors

Crew: Director(s)

Overview: Plot summary

🎯 Future Improvements
User authentication with personalized watchlist

Collaborative filtering suggestions

More advanced filtering options (release year, runtime)

Movie trailer integration

User ratings and reviews

Mobile responsive design improvements

Dark/Light theme toggle

Export recommendations to CSV

🤝 Contributing
Contributions are welcome! Follow these steps:

Fork the repository

Create a feature branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add some AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Karan Farne

https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white
https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white

🙏 Acknowledgments
Data provided by TMDB (The Movie Database)

Built with Streamlit

Inspired by content-based filtering research

Special thanks to the open-source community

⭐ Support
If this project helped you, please give it a star on GitHub! ⭐

📧 Contact
For any questions or suggestions, feel free to reach out:

GitHub: karanfarane-prog

LinkedIn: Karan Farane
