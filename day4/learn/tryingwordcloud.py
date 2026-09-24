# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Provided text
text = """
The Guitar
The guitar is one of the most popular musical instruments in the world. It is a string instrument that usually has six strings and can be played with the fingers or a pick. Guitars are made in different shapes and sizes, and they are commonly divided into acoustic, classical, and electric guitars.
The guitar has a beautiful and versatile sound. It is used in many types of music, including classical, rock, pop, jazz, blues, country, and folk music. A guitarist can play both melodies and chords, which makes the instrument suitable for solo performances as well as playing in a band.
Learning to play the guitar requires regular practice, patience, and concentration. Beginners usually start by learning simple chords and basic songs. With time and practice, they can learn more difficult techniques and develop their own style.
The guitar is not only a source of entertainment but also a wonderful way to express feelings and creativity. Playing it can be relaxing and enjoyable. It can also improve concentration and confidence. For these reasons, the guitar remains a favorite instrument among people of all ages.
"""

print(text)

# Tokenization of words
words = word_tokenize(text.lower())
print(words)

# Remove stopwords and non-alphabetic words
stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in words
    if word.isalpha() and word not in stop_words
]

print(filtered_words)

# Count the frequency of each word
word_freq = Counter(filtered_words)

print(word_freq)

# Create a WordCloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate_from_frequencies(word_freq)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
