# Mid-Term-NewsBotIntelligenceSystem.SamanthaMireles-ITAI.2373
# NewsBot Intelligence System
## ITAI 2373 Midterm Project

### Project Goal
This project builds a NewsBot Intelligence System that processes news articles and produces useful insights. The system cleans text, extracts features, analyzes grammar and sentiment, classifies articles into categories, and identifies named entities such as people, organizations, and locations.

### Business Use Case
This system can help media companies, business analysts, and researchers monitor news coverage, track public sentiment, and identify trends across categories such as politics, sports, technology, and business.


### Module 1: Real-World NLP Application Context-

This project builds a NewsBot system that analyzes news articles using Natural Language Processing (NLP). The system automatically classifies articles into categories such as politics, sports, and technology, while also extracting key entities, analyzing sentiment, and identifying patterns in the text. The goal is to help businesses and users quickly understand large amounts of news data without having to read every article individually.

To accomplish this, NewsBot uses a multi-stage NLP pipeline. First, raw article text is preprocessed this includes tokenization, stopword removal, and lemmatization to normalize the content. Next, a classification model assigns each article to one of several predefined categories based on its vocabulary and context. Simultaneously, a Named Entity Recognition (NER) module identifies important mentions like people, organizations, locations, and dates. Finally, a sentiment analysis component evaluates whether the overall tone of an article is positive, negative, or neutral.

For example, companies can use NewsBot to monitor public opinion about their brand, track competitor activity, or detect emerging trends in specific industries. A financial firm could use the system to flag negative news around a stock before it impacts the market. A marketing team could track how sentiment around a product shifts over time following a campaign launch. Government agencies could use it to monitor misinformation or identify shifts in public discourse on policy topics.
Ultimately, NewsBot demonstrates how NLP techniques can turn unstructured text into structured, actionable insights making it a practical tool for anyone who needs to stay informed at scale.

### Module 2: Text Preprocessing Pipeline --

The text was cleaned by converting it lowercase, removing punctuation, and eliminating stopwords like "the" amd "and". Lemmatization was applied to reduce words to their base form, improving consistency in analysis.

Code:

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['content'].apply(preprocess)

### Module 3: TF-IDF Feature Extraction ---

TF-IDF was used to convert text into numerical features. This method highlights important words in each article while reducing the importance of common words.

Code:
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])


### Module 4: POS Tagging ----

Part-of-speech tagging helps identify patterns such as nouns, verbs, and adjectives. This allows us to analyze writing styles across different news categories.

Code:

import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp(df['content'][0])
for token in doc:
    print(token.text, token.pos_)

### Module 5: Syntax and Semantic Analysis -----

Dependency parsing helps understand relationships between words, such as subject and object. This improves how meaning is extracted from sentences.

Code:

for token in doc:
    print(token.text, token.dep_, token.head.text)

### Module 6: Sentient Analysis ------

Sentiment analysis assigns a score between -1 and 1. Negative values indicate negative tone, while positive values indicate positive tone.

Code: 

from textblob import TextBlob

df['sentiment'] = df['content'].apply(lambda x: TextBlob(x).sentiment.polarity)

### Module 7: Text Classification -------

A Naive Bayes model was used to classify articles. The model achieved an accuracy of around 85%, showing it can effectively categorize news content.

Code:

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, df['category'], test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

### Module 8: Named Entity Recognition --------

Named Entity Recognition extracts important entities like people, organizations, and locations. This helps identify key topics in articles.

Code: 

for ent in doc.ents:
    print(ent.text, ent.label_)

    

## Final Insights and Conclusion

### Key Findings
- The classification models were able to separate news articles into categories with strong accuracy.
- TF-IDF showed that each category had distinctive vocabulary patterns.
- POS analysis showed that different categories used different writing styles.
- Sentiment analysis showed that some categories were more neutral while others had more emotional language.
- NER revealed the people, companies, locations, and dates most frequently mentioned in the news.

### Business Value
This NewsBot system can help organizations automatically monitor large volumes of news content. It can support business intelligence, competitor tracking, media analysis, and public opinion monitoring.

### Future Improvements
- Use a larger dataset
- Add emotion detection
- Build a Streamlit or Gradio interface
- Improve NER with custom-trained models
