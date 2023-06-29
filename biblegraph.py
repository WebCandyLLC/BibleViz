import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('gutenberg')
#nltk.download('vader_lexicon')
#from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist
from nltk import bigrams
import spacy
import numpy as np
import argparse
import re
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
from wordcloud import WordCloud
import seaborn as sns
import networkx as nx

# NOTES:
# Maybe consider doing a sentence scope with the sentiment analysis.
# Also consider labelling the sentiment plot with the chapter label rather than word count label
# Or provide a function that allows for word count position to detail which chapter and verse in the book it is from.

def classify_words(sentences):
    causation_patterns = ['because of', 'due to the fact', 'in view of', 'as a result of', 'on account of', 'owing to', 'as a consequence of', 'as a result', 'therefore', 'thus', 'hence', 'consequently', "because"]
    substantiation_patterns = ['for example', 'for instance', 'such as', 'including', 'like', 'e.g.', 'i.e.', 'namely', "for"]
    contrast_patterns = ['on the other hand', 'in contrast', 'while', 'although', 'though', 'but', 'however', 'yet', 'despite', 'in spite of', 'regardless', 'nonetheless', 'nevertheless']
    comparative_patterns = ['in the same way', 'similarly', 'likewise', 'equally', 'comparable to', 'in comparison', 'as', 'just as', 'as... as', 'as if', 'as though']
    interrogation_patterns = ['?']

    classifications = {
        'Causation': [],
        'Substantiation': [],
        'Contrast': [],
        'Comparative': [],
        'Interrogation': []
    }

    for i, sentence in enumerate(sentences):
        for pattern in causation_patterns:
            matches = re.finditer(r'\b' + re.escape(pattern) + r'\b', sentence, re.IGNORECASE)
            positions = [match.start() for match in matches]
            if positions:
                classifications['Causation'].append([i, positions])

        for pattern in substantiation_patterns:
            matches = re.finditer(r'\b' + re.escape(pattern) + r'\b', sentence, re.IGNORECASE)
            positions = [match.start() for match in matches]
            if positions:
                classifications['Substantiation'].append([i, positions])

        for pattern in contrast_patterns:
            matches = re.finditer(r'\b' + re.escape(pattern) + r'\b', sentence, re.IGNORECASE)
            positions = [match.start() for match in matches]
            if positions:
                classifications['Contrast'].append([i, positions])

        for pattern in comparative_patterns:
            matches = re.finditer(r'\b' + re.escape(pattern) + r'\b', sentence, re.IGNORECASE)
            positions = [match.start() for match in matches]
            if positions:
                classifications['Comparative'].append([i, positions])
        
        for pattern in interrogation_patterns:
            matches = re.finditer(r'\?', sentence, re.IGNORECASE)
            positions = [match.start() for match in matches]
            if positions:
                classifications['Interrogation'].append([i, positions])

    return classifications

def verse_tokenize(contents):
    return [item for v in contents.values() for item in v]

def sentence_tokenize(contents):
    # Tokenize the string into sentences
    return sent_tokenize(contents)

def words_tokenize(sentences):
    # Tokenize each sentence into words
    words = []
    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        # Filter out non-alphanumeric characters
        sentence_words = [word for word in sentence_words if re.match(r'^[a-zA-Z0-9]+$', word)]
        words.extend(sentence_words)
    return words

def get_bible_contents(book, start_chapter, start_verse, end_chapter, end_verse):
    file_name = "outputV2.json"  # Update with the actual file name

    # Read the JSON file
    with open(file_name, "r") as file:
        data = json.load(file)

    book_data = data.get(book)

    if book_data is None:
        print("Invalid book selection")
        return None
    
    num_chapters = len(book_data)

    if not (1 <= start_chapter <= num_chapters):
        print(f"Invalid starting chapter: {start_chapter}")
        return None

    if not (start_chapter <= end_chapter <= num_chapters):
        print(f"Invalid ending chapter: {end_chapter}")
        return None
    start_chapter_data = book_data[str(start_chapter)]["verses"]
    num_verses_start = len(start_chapter_data)

    if not (1 <= start_verse <= num_verses_start):
        print(f"Invalid starting verse: {start_verse}")
        return None

    end_chapter_data = book_data[str(end_chapter)]["verses"]
    num_verses_end = len(end_chapter_data)

    if not (1 <= end_verse <= num_verses_end):
        print(f"Invalid ending verse: {end_verse}")
        return None

    data = {}

    contents = ""

    for chapter in range(start_chapter, end_chapter + 1): 
        chapter_data = book_data.get(str(chapter))

        if chapter_data is None:
            print(f"Invalid chapter selection: {chapter}")
            continue

        if chapter == end_chapter:
            currentEndVerse = end_verse + 1
        else:
            currentEndVerse = len(chapter_data["verses"])
        
        if chapter == start_chapter:
            currentStartVerse = start_verse
        else:
            currentStartVerse = 1

        data[str(chapter)] = []

        for verse in range(currentStartVerse, currentEndVerse):

            verse_content = chapter_data["verses"][verse - 1]

            if verse_content is None:
                print(f"Invalid verse selection: {chapter}:{verse}")
                continue

            data[str(chapter)].append(verse_content)
            #contents += verse_content + " "

    return data
    #return contents.strip()

def stringify_bible_contents(content):
    contents = ""

    for k, v in content.items(): 
        chapter_data = content.get(k)

        for verse in v:
            verse_content = verse
            contents += verse_content + " "

    return contents.strip()

def perform_co_occurrence_analysis(tokens):
    # Create token index mapping
    unique_tokens = list(set(tokens))
    token_index_map = {token: index for index, token in enumerate(unique_tokens)}

    # Calculate co-occurrence matrix
    token_pairs = list(bigrams(tokens))
    co_occurrence_matrix = np.zeros((len(unique_tokens), len(unique_tokens)), dtype=int)

    # Calculate frequencies of token pairs
    fdist = FreqDist(token_pairs)
    for (token1, token2), frequency in fdist.items():
        token1_idx = token_index_map[token1]
        token2_idx = token_index_map[token2]
        co_occurrence_matrix[token1_idx][token2_idx] = frequency

    return co_occurrence_matrix

def perform_network_analysis(tokens):
    # Load the spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Perform named entity recognition
    doc = nlp(" ".join(tokens))
    entities = [entity.text for entity in doc.ents]

    # Create the network graph
    edges = []
    for token in doc:
        if token.ent_type_ != "":
            edges.append((token.text, token.ent_type_))

    return edges

def perform_sentiment_analysis(items):
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiment scores for each token
    sentiment_scores = []
    for item in items:
        sentiment_score = analyzer.polarity_scores(item)
        sentiment_scores.append(sentiment_score)

    return sentiment_scores

def visualize_n_freqs(words):

    # Calculate word frequencies for each word length
    word_length = 1
    while True:
        filtered_words = []
        for i in range(len(words) - word_length + 1):
            n_word = ' '.join(words[i:i+word_length])
            filtered_words.append(n_word)

        # Apply stop word removal only for 1-word frequency analysis
        if word_length == 1:
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in filtered_words if word.lower() not in stop_words]

        # Apply case-insensitive filtering for specific words
        filtered_words = [word if word in ['God', 'Jesus', 'Christ', 'LORD'] else word.lower() for word in filtered_words]
        word_frequencies = nltk.FreqDist(filtered_words)

        # Check if all frequencies are 1 for the current word length
        if all(frequency == 1 for frequency in word_frequencies.values()):
            break

        # Filter and plot the word frequency if frequencies are not all 1
        filtered_frequencies = {word: freq for word, freq in word_frequencies.items() if freq > 1}
        sorted_frequencies = sorted(filtered_frequencies.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_frequencies) > 15:
            sorted_frequencies = sorted_frequencies[:15]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{word_length}-Word Frequencies')

        if word_length > 9:
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')
            ax.yaxis.get_major_locator().set_params(integer=True)

            color = [(random.random(), random.random(), random.random()) for _ in range(len(sorted_frequencies))]
            labels = ['' if freq <= 1 else word for word, freq in sorted_frequencies]
            values = [freq for word, freq in sorted_frequencies]
            bars = ax.bar(labels[:15], values[:15], color=color[:15])
            legend_labels = [f'{word} ({freq})' for word, freq in sorted_frequencies if freq > 1]
            ax.legend(bars, legend_labels, loc='upper right')

        else:
            labels, values = zip(*sorted_frequencies)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.tick_params(axis='x', which='major', pad=10)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            ax.bar(range(len(labels)), values)
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')
            
        plt.tight_layout()
        plt.show()

        word_length += 1

def visualize_inductive(sentences):
    #Defining the classification symbols print out.
    symbols = {
        'Causation': '⇒',
        'Substantiation': '⇐',
        'Contrast': '©',
        'Comparative': '⇔',
        'Interrogation': '?'
    }

    # Collect classifications for each sentence
    classifications = classify_words(sentences)

    # Plot sentence classifications
    sentences_per_plot = 6
    num_sentences = len(sentences)
    num_plots = math.ceil(num_sentences / sentences_per_plot)

    for plot_num in range(num_plots):
        start_idx = plot_num * sentences_per_plot
        end_idx = min((plot_num + 1) * sentences_per_plot, len(sentences))
        plot_sentences = sentences[start_idx:end_idx]
        plot_classifications = {
            classification: [
                [sentence_idx, positions]
                for sentence_idx, positions in positions_list
                if sentence_idx >= start_idx and sentence_idx < end_idx
            ]
            for classification, positions_list in classifications.items()
        }

        # Create subplots for the current plot
        num_subplots = len(plot_sentences)
        fig, axs = plt.subplots(num_subplots, 1, figsize=(12, num_subplots))
        fig.subplots_adjust(hspace=0.5)

        for i, ax in enumerate(axs):
            sentence_idx = start_idx + i
            ax.set_title(f'Sentence {sentence_idx+1}')
            ax.title.set_fontsize(10)  # Set the desired font size
            ax.set_yticks([])
            sentence_text = plot_sentences[i]
            ax.set_xticks(range(len(sentence_text)))
            ax.set_xticklabels(sentence_text)

            # Get the y-axis limits based on the positions of classification symbols
            ymin, ymax = ax.get_ylim()

            for classification, positions in plot_classifications.items():
                for position in positions:
                    if position[0] == sentence_idx:
                        for pos in position[1]:
                            ax.text(pos, ymin + (ymax - ymin) / 2, symbols[classification], ha='center', va='center')

            # Set the y-axis limits to match the height of classification symbols
            ax.set_ylim(ymin, ymax)

        plt.tight_layout()
        plt.show()

def visualize_word_cloud(tokens):
    wordcloud = WordCloud().generate(' '.join(tokens))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def visualize_histogram(tokens):
    word_lengths = [len(token) for token in tokens]
    plt.hist(word_lengths, bins=10)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.show()

def visualize_heatmap(co_occurrence_matrix):
    sns.heatmap(co_occurrence_matrix, cmap='YlOrRd')
    plt.xlabel('Words')
    plt.ylabel('Words')
    plt.show()

def visualize_network_graph(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True)
    plt.show()

def visualize_sentiment_plot(sentiment_scores):
    # Prepare the data for plotting
    x = list(range(len(sentiment_scores)))
    positive_scores = [score['pos'] for score in sentiment_scores]
    negative_scores = [score['neg'] for score in sentiment_scores]
    neutral_scores = [score['neu'] for score in sentiment_scores]

    # Create stacked area chart
    plt.stackplot(x, positive_scores, negative_scores, neutral_scores, labels=['Positive', 'Negative', 'Neutral'],
                  colors=['green', 'red', 'blue'])

    # Set plot properties
    plt.xlabel('Index')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis')
    plt.legend(loc='upper left')

    # Show the plot
    plt.show()

def process_book_range():
    parser = argparse.ArgumentParser(description='Process book range and perform visualization.')
    parser.add_argument('book', type=str, help='Path to the book file')
    parser.add_argument('start_chapter', type=int, help='Starting chapter')
    parser.add_argument('start_verse', type=int, help='Starting verse')
    parser.add_argument('end_chapter', type=int, help='Ending chapter')
    parser.add_argument('end_verse', type=int, help='Ending verse')
    parser.add_argument('--n_freqs', action='store_true', help='Generate the plot(s) N-Word Frequencies')
    parser.add_argument('--inductive', action='store_true', help='Generate the inductive observation markings')
    parser.add_argument('--word_cloud', action='store_true', help='Generate a word cloud')
    parser.add_argument('--histogram', action='store_true', help='Generate a histogram of word length frequencies.')
    parser.add_argument('--heatmap', action='store_true', help='Generate a heatmap to visualize the co-occurrence or similarity matrix between words.')
    parser.add_argument('--network_graph', action='store_true', help='Generate a network graph to visualize the connections between word entities.')
    parser.add_argument('--sentiment', action='store_true', help='Generate a sentiment plot')
    parser.add_argument('--sentence', action='store_true', help='Uses the sentence scope with the visualization')
    parser.add_argument('--verses', action='store_true', help='Uses the verse scope with the visualization')
    
    args = parser.parse_args()
    book = args.book
    start_chapter = args.start_chapter
    start_verse = args.start_verse
    end_chapter = args.end_chapter
    end_verse = args.end_verse

    visualization = None
    if args.n_freqs:
        visualization = 'n_freqs'
    elif args.inductive:
        visualization = 'inductive'
    elif args.word_cloud:
        visualization = 'word_cloud'
    elif args.histogram:
        visualization = 'histogram'
    elif args.heatmap:
        visualization = 'heatmap'
    elif args.network_graph:
        visualization = 'network_graph'
    elif args.sentiment:
        visualization = 'sentiment'

    if visualization:
        # Here you can perform any processing or analysis using the provided book and verse range
        print(f"Processing book: {book}")
        print(f"Verse range: {start_chapter}:{start_verse} - {end_chapter}:{end_verse}")

        bible_contents_dict = get_bible_contents(book, start_chapter, start_verse, end_chapter, end_verse)
        bible_contents = stringify_bible_contents(bible_contents_dict)

        if bible_contents:
            if args.verses:
                verses = verse_tokenize(bible_contents_dict)
            if not args.verses:
                sentences = sentence_tokenize(bible_contents)

                if not args.sentence:
                    tokens = words_tokenize(sentences)

            # Perform visualization based on the specified flag
            if visualization == 'n_freqs':
                # Perform the N-Freq plotting where N is the number of words in the phrase.
                visualize_n_freqs(tokens)
            elif visualization == 'inductive':
                # Perform the Inductive Plotting of the Observation Markers.
                visualize_inductive(sentences)
            elif visualization == 'word_cloud':
                visualize_word_cloud(tokens)
            elif visualization == 'histogram':
                visualize_histogram(tokens)
            elif visualization == 'heatmap':
                # Perform co-occurrence analysis and pass the co-occurrence matrix to visualize_heatmap function
                co_occurrence_matrix = perform_co_occurrence_analysis(tokens)
                visualize_heatmap(co_occurrence_matrix)
            elif visualization == 'network_graph':
                # Perform network analysis and pass the edges to visualize_network_graph function
                edges = perform_network_analysis(tokens)
                visualize_network_graph(edges)
            elif visualization == 'sentiment':
                if args.sentence:
                    sentiment_scores = perform_sentiment_analysis(sentences)
                elif args.verses:
                    sentiment_scores = perform_sentiment_analysis(verses)
                else:
                    # Perform sentiment analysis and pass the sentiment scores to visualize_sentiment_plot function
                    sentiment_scores = perform_sentiment_analysis(tokens)
                visualize_sentiment_plot(sentiment_scores)
            else:
                print("Invalid visualization method!")
        else:
            print('Bible Range selection invalid!')
    else:
        print("No visualization method specified!")

# Example usage:
# python my_script.py Genesis 1 1 3 16 <visualization method flag>
if __name__ == '__main__':
    process_book_range()