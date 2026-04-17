# Neural Collaborative Filtering - Movie Recommendation System

A deep learning-based recommendation system developed as a Master's thesis in Software Engineering.  
The model learns user and movie latent representations via embedding layers and predicts personalized ratings using a neural network.

---

## How It Works

The model uses **Neural Collaborative Filtering (NCF)** - instead of classical matrix factorization, user-item interactions are learned by a neural network that can capture non-linear patterns.

Each user and each movie is mapped to a dense embedding vector (latent factors). These vectors are concatenated and passed through fully connected layers to predict the rating a user would give a movie.

```
User ID  ──► Embedding(n_users, 10)  ──► Flatten ──► Dropout(0.2) ──┐
                                                                       ├──► Concat ──► FC(100) ──► FC(50) ──► FC(20) ──► FC(10, ReLU) ──► Output(1, ReLU)
Movie ID ──► Embedding(n_movies, 13) ──► Flatten ──► Dropout(0.2) ──┘
```

**Key design decisions:**
- Separate embedding layers: 10 latent factors for users, 13 for movies
- Dropout(0.2) after each major block to prevent overfitting
- Dense tower: 100 → 50 → 20 → 10 → 1
- Optimizer: Adam, lr=1e-5
- Loss: Mean Absolute Error (MAE)
- Early stopping monitored during training via validation split (10%)

---

## Dataset

Three CSV files from [MovieLens](https://grouplens.org/datasets/movielens/):

| File | Description |
|------|-------------|
| `movies.csv` | Movie IDs and titles |
| `movies_metadata.csv` | Extended movie metadata |
| `movies_ratings.csv` | User ratings (userId, movieId, rating, timestamp) |

Train/test split: **75% / 25%**

User and movie IDs are re-encoded as contiguous integer categories before training.

---

## Results

| Metric | Value |
|--------|-------|
| MAE on test set | **0.673** |

Training and validation loss were tracked over 30 epochs. Predictions on the test set are rounded to the nearest 0.5 (matching MovieLens rating scale).

---

## Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Keras / TensorFlow | Model definition and training |
| pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| scikit-learn | Train/test split, MAE metric |
| Google Colab | Training environment |
| Matplotlib | Loss curve visualization |

---

## Project Structure

```
Neural-Collaborative-Filtering/
├── Dyploma_CNN_RS.ipynb   # Full pipeline: data loading, model, training, evaluation
├── datasets/
│   ├── movies.csv
│   ├── movies_metadata.csv
│   └── movies_ratings.csv
└── README.md
```

---

## How to Run

The notebook was originally developed in Google Colab with data stored on Google Drive.  
To run locally:

```bash
git clone https://github.com/ArtemMint/Neural-Collaborative-Filtering.git
cd Neural-Collaborative-Filtering

pip install tensorflow keras pandas numpy scikit-learn matplotlib jupyter
jupyter notebook Dyploma_CNN_RS.ipynb
```

Update the `BASE_PATH` variable in the notebook to point to your local dataset directory:

```python
# Replace this:
BASE_PATH = "/content/gdrive/My Drive/Dyploma/"

# With your local path:
BASE_PATH = "./datasets/"
```

---

## Background

Developed as a Master's thesis in Software Engineering.  
The project explores neural embedding-based approaches to collaborative filtering as an alternative to classical matrix factorization methods.

---

## References

- [Neural Collaborative Filtering (2017)](https://arxiv.org/abs/1708.05031)
- [MovieLens Dataset — GroupLens](https://grouplens.org/datasets/movielens/)
