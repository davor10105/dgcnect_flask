import random
from cleantext import clean
from simplemma import simple_tokenizer, lemmatize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from multiprocessing import Pool
from model_data import TenderData, CountryModelData, LanguageModelData


RANDOM_SEED = 69


class Trainer:
    def return_input(example, language):
        text = example[3] + example[4]
        text = clean(
            text,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=False,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
            no_urls=False,  # replace all URLs with a special token
            no_emails=False,  # replace all email addresses with a special token
            no_phone_numbers=False,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_punct="",  # instead of removing punctuations you may replace them
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en",  # set to 'de' for German special handling
        )

        tokens = simple_tokenizer(text)
        lemmatized_tokens = [lemmatize(token, lang=language) for token in tokens]
        return tokens, lemmatized_tokens

    def train(dataset, language, stop_words=[]):
        examples = []
        print("Cleaning data...")
        for example in tqdm(dataset):
            tokens, lemmatized_tokens = Trainer.return_input(example, language)
            examples.append(
                {
                    "original": " ".join(tokens),
                    "input_text": " ".join(lemmatized_tokens),
                    "label": int(example[5]),
                    "tender_id": str(example[7]),
                }
            )

        train_ratio = 0.8
        random.seed(RANDOM_SEED)
        random.shuffle(examples)

        num_train = int(len(examples) * train_ratio)
        train_examples = examples[:num_train]
        test_examples = examples[num_train:]

        train_texts = [example["input_text"] for example in train_examples]
        train_labels = np.array([example["label"] for example in train_examples])
        test_texts = [example["input_text"] for example in test_examples]
        test_labels = np.array([example["label"] for example in test_examples])

        if len(stop_words) == 0:
            print("Obtaining stop words...")
            vectorizer = TfidfVectorizer(max_df=0.05)
            _ = vectorizer.fit_transform(train_texts)
            stop_words = list(vectorizer.stop_words_)

        print("Training model...")
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        clf = LogisticRegression(
            random_state=RANDOM_SEED, class_weight="balanced", C=0.6
        ).fit(train_features, train_labels)

        all_texts = [example["input_text"] for example in examples]
        all_tender_ids = [example["tender_id"] for example in examples]
        all_labels = np.array([example["label"] for example in examples])
        all_features = vectorizer.transform(all_texts)
        all_preds = clf.predict(all_features)

        tender_data = TenderData(all_features, all_preds, all_labels, all_tender_ids)
        language_model_data = LanguageModelData(
            clf, vectorizer, stop_words, tender_data
        )
        print("Success")

        return language_model_data
