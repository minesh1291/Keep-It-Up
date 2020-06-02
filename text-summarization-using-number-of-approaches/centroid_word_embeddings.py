"""
  First install text_summarizer using
  pip install git+https://github.com/lambdaofgod/text-summarizer
"""

import nltk
import text_summarizer

# prepare nltk data
nltk.download('punkt')
nltk.download('stopwords')

text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text

# we'll need embedding model from gensim for summarizer
# this can take a while
embedding_model = text_summarizer.centroid_word_embeddings.load_gensim_embedding_model('glove-wiki-gigaword-50')

centroid_word_embedding_summarizer = text_summarizer.CentroidWordEmbeddingsSummarizer(embedding_model, preprocess_type='nltk')

centroid_word_embedding_summary = centroid_word_embedding_summarizer.summarize(text)
