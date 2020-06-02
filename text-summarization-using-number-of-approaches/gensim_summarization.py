import nltk
import gensim

text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt').text

gensim_summary = gensim.summarization.summarize(text)
