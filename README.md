# Taylor Swift Lyrics: Discography Modeling Project

## Overview
This project analyzes Taylor Swift lyrics with NLP, classical ML, and embedding-based neural workflows to surface discography-level patterns.

Primary notebook:
- `TS .ipynb`

Primary dataset:
- `taylor_swift_lyrics.csv`

The notebook combines descriptive analysis, topic modeling, clustering, classification, and similarity graphing to answer questions like:
- How do lyrical themes vary by album/era?
- Which albums are lyrically closest to each other?
- Which songs are most similar across the discography?

## Tech Stack
- Python 3
- Jupyter Notebook
- pandas, numpy, matplotlib
- scikit-learn
- sentence-transformers
- umap-learn
- nltk
- spaCy (`en_core_web_sm`)
- networkx (for graph visualizations)

## Setup
1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn sentence-transformers umap-learn nltk spacy networkx
python -m spacy download en_core_web_sm
```

3. Ensure `taylor_swift_lyrics.csv` is in the same folder as `TS .ipynb`.

## Project Workflow
### 1) Base NLP + exploratory analysis
- Text cleaning and line/song aggregation
- Lexical features (e.g., type-token style metrics)
- Sentiment-oriented line/song views

### 2) TF-IDF + interpretable models
- `TfidfVectorizer` at song level
- NMF topic modeling with tuning for topic stability
- KMeans clustering of songs
- Logistic Regression for album prediction

### 3) Embedding + neural extension
- Song embeddings via `SentenceTransformer('all-MiniLM-L6-v2')`
- Album prediction with `MLPClassifier`
- 2D embedding visualization (UMAP)

### 4) Discography correlation analysis
- Album centroid cosine similarity matrix + heatmap
- Era drift plot: year vs similarity to earliest-album style anchor

### 5) Similarity tools
- `plot_similar_songs(...)`: ranked and plotted nearest songs
- `plot_similarity_network(...)`: local graph of query song + neighbors

## Key Functions Added for Similarity/Modeling
- `plot_similar_songs(track_title, album=None, top_k=8, exclude_same_album=False)`
- `plot_similarity_network(track_title, album=None, top_k=12, neighbor_edge_threshold=0.55, exclude_same_album=False)`

## How to Run
1. Launch Jupyter from this folder:
   - `jupyter notebook`
2. Open `TS .ipynb`.
3. Run all cells from top to bottom.
4. Re-run the final similarity cells with different query songs/albums for targeted comparisons.

## Example Calls
```python
plot_similar_songs('Style', top_k=10, exclude_same_album=True)
plot_similarity_network('All Too Well', top_k=12, neighbor_edge_threshold=0.58)
```

## Notes
- Embedding and neural sections may take longer on first run due to model loading.
- Similarity outputs depend on preprocessing choices (stopwords, n-gram settings, min/max document frequency).
