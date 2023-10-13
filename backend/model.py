import matplotlib.pyplot as plt
import pickle
import sklearn
import os
import numpy as np
import uuid
import io
import psycopg2
import trainer
import pickle
from database_login import DBNAME, USER, PASSWORD, HOST, PORT


alpha2name = {
    "UK": "United Kingdom",
    "DE": "Germany",
    "HR": "Croatia",
    "AT": "Austria",
    "NL": "The Netherlands",
    "IT": "Italy",
    "FR": "France",
}

# map country to language tokenizer
country2language = {
    "HR": "hbs",
}


class PostgresCountryModel:
    def __init__(self) -> None:
        self.conn = psycopg2.connect(
            dbname=DBNAME,
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
        )
        self.cur = self.conn.cursor()
        self.cur.execute("select distinct country_iso from dataset")
        countries = self.cur.fetchall()
        countries = [country[0] for country in countries]

        if not os.path.exists("data"):
            os.makedirs("data")

        for country in countries:
            saved_path = os.path.join("data", f"{country}.pickle")
            if not os.path.exists(saved_path):
                country_dataset = self.fetch_dataset(country)
                model_data = trainer.Trainer.train(
                    country_dataset, country2language[country]
                )

                with open(f"data/{country}.pickle", "wb") as f:
                    pickle.dump(model_data, f)

        country_model_data = {}
        for country in countries:
            with open(f"data/{country}.pickle", "rb") as f:
                model_data = pickle.load(f)
            country_model_data[country] = model_data
        self.country_model_data = country_model_data

        self.detailed_country_data = {}

        self.global_data = {}
        for country in self.country_model_data:
            score_key = []
            (
                clf,
                vectorizer,
                _,
                _,
            ) = self.country_model_data[country]
            for key, index in vectorizer.vocabulary_.items():
                score_key.append((key, clf.coef_[0][index]))
            score_key = sorted(score_key, reverse=True, key=lambda k: k[1])
            self.global_data[country] = score_key

    def fetch_dataset(self, country):
        self.cur.execute(f"SELECT * FROM dataset where country_iso='{country}'")
        dataset = self.cur.fetchall()

        return dataset

    def get_countries_data(self):
        retval = []
        for key in self.country_model_data.keys():
            retval.append({"CountryName": alpha2name[key], "Country2Alpha": key})
        return retval

    def calculate_details_for_country(self, country):
        (
            clf,
            vectorizer,
            (all_features, all_preds, all_labels),
            examples,
        ) = self.country_model_data[country]
        metadata_dict = {
            "NumExamples": all_preds.shape[0],
            "NumInnovative": all_labels.sum().item(),
            "NumNonInnovative": (all_preds.shape[0] - all_labels.sum()).item(),
        }
        selected_prediction_type_dict = {
            "TruePositive": [],
            "TrueNegative": [],
            "FalsePositive": [],
            "FalseNegative": [],
        }
        for i, (all_label, all_pred) in enumerate(zip(all_labels, all_preds)):
            if all_label == 1:
                if all_label == all_pred:
                    selected_prediction_type_dict["TruePositive"].append(str(i))
                else:
                    selected_prediction_type_dict["FalseNegative"].append(str(i))
            else:
                if all_label == all_pred:
                    selected_prediction_type_dict["TrueNegative"].append(str(i))
                else:
                    selected_prediction_type_dict["FalsePositive"].append(str(i))
        self.detailed_country_data[country] = {
            "Metadata": metadata_dict,
            "Details": selected_prediction_type_dict,
        }

    def get_country_data(self, country):
        self.calculate_details_for_country(country=country)
        return self.detailed_country_data[country]

    def get_global_data(self, country):
        return {
            "TopWords": self.global_data[country][:100],
            "BottomWords": self.global_data[country][-100:][::-1],
        }

    def get_tender_data(self, country, tender_id):
        (
            clf,
            vectorizer,
            (all_features, all_preds, all_labels),
            examples,
        ) = self.country_model_data[country]

        tender_index = int(tender_id)

        word_scores = all_features[tender_index].multiply(clf.coef_[0]).tocsr()
        scored_words = []
        total_score = all_features[tender_index].dot(clf.coef_[0])
        word_score = {}
        lemma_original = {}
        for i, (original_word, lemma_word) in enumerate(
            zip(
                examples[tender_index]["original"].split(" "),
                examples[tender_index]["input_text"].split(" "),
            )
        ):
            score = 0.0
            if lemma_word in vectorizer.vocabulary_:
                word_index = vectorizer.vocabulary_[lemma_word]
                score = word_scores[0, word_index].item()

            if lemma_word not in word_score:
                word_score[lemma_word] = 0
            if lemma_word not in lemma_original:
                lemma_original[lemma_word] = []
            word_score[lemma_word] = score
            lemma_original[lemma_word].append(original_word)

            scored_words.append([original_word, score])

        fig, ax = plt.subplots()
        word_score = dict(sorted(word_score.items(), key=lambda k: k[1]))
        vis_words = []
        current_sum = 0
        bias = np.array(clf.intercept_)
        ax.axvline(x=-bias[0], color="red", label="decision boundary")
        ax.text(
            -bias[0] + 0.05, 5, "decision boundary", rotation=90, color="r", va="center"
        )
        ax.axvline(x=0, color="black", label="zero", linestyle="dashed")

        top_keys = list(word_score.keys())[-5:]
        top_keys.reverse()
        for i, lemma_word in enumerate(top_keys):
            # if word_score[lemma_word] == 0:
            #    continue
            ax.barh(i, current_sum + word_score[lemma_word], align="center", color="r")
            ax.barh(i, current_sum, align="center", color="white")
            current_sum += word_score[lemma_word]
            vis_words.append(lemma_original[lemma_word][0].lower())
            ax.text(
                current_sum + 0.1,
                i,
                str(round(word_score[lemma_word], 2)),
                color="r",
                va="center",
            )

        bot_keys = list(word_score.keys())[:5]
        # bot_keys.reverse()
        for i, lemma_word in enumerate(bot_keys):
            # if word_score[lemma_word] == 0:
            #    continue
            if current_sum > 0:
                zero_to_pos = current_sum + word_score[lemma_word]
                ax.barh(i + 5, current_sum, align="center", color="b")
                if zero_to_pos > 0:
                    ax.barh(i + 5, zero_to_pos, align="center", color="white")
                else:
                    ax.barh(i + 5, current_sum, align="center", color="b")
                    ax.barh(i + 5, zero_to_pos, align="center", color="b")
            else:
                ax.barh(
                    i + 5,
                    current_sum + word_score[lemma_word],
                    align="center",
                    color="b",
                )
                ax.barh(i + 5, current_sum, align="center", color="white")
            ax.text(
                current_sum + 0.1,
                i + 5,
                str(round(word_score[lemma_word], 2)),
                color="blue",
                va="center",
            )
            current_sum += word_score[lemma_word]
            vis_words.append(lemma_original[lemma_word][0].lower())
        ax.set_yticks(np.linspace(0, 10, 10), labels=vis_words)
        ax.invert_yaxis()
        fig.tight_layout()

        filename = uuid.uuid4()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_image = buf.read().hex()

        return {"WordScores": scored_words, "Plot": byte_image}


class CountryModel:
    def __init__(self) -> None:
        countries = [filename[:2].upper() for filename in os.listdir("data")]
        country_model_data = {}
        for country in countries:
            with open(f"data/{country.lower()}.pickle", "rb") as f:
                model_data = pickle.load(f)
            country_model_data[country] = model_data
        self.country_model_data = country_model_data

        self.detailed_country_data = {}

        self.global_data = {}
        for country in self.country_model_data:
            score_key = []
            (
                clf,
                vectorizer,
                (test_features, test_pred),
                test_metrics,
                (test_original_texts, test_texts, test_labels),
            ) = self.country_model_data[country]
            for key, index in vectorizer.vocabulary_.items():
                score_key.append((key, clf.coef_[0][index]))
            score_key = sorted(score_key, reverse=True, key=lambda k: k[1])
            self.global_data[country] = score_key

    def get_countries_data(self):
        retval = []
        for key in self.country_model_data.keys():
            retval.append({"CountryName": alpha2name[key], "Country2Alpha": key})
        return retval

    def calculate_details_for_country(self, country):
        (
            clf,
            vectorizer,
            (test_features, test_pred),
            test_metrics,
            (test_original_texts, test_texts, test_labels),
        ) = self.country_model_data[country]
        metadata_dict = {
            "NumExamples": test_pred.shape[0],
            "NumInnovative": test_labels.sum().item(),
            "NumNonInnovative": (test_pred.shape[0] - test_labels.sum()).item(),
        }
        selected_prediction_type_dict = {
            "TruePositive": [],
            "TrueNegative": [],
            "FalsePositive": [],
            "FalseNegative": [],
        }
        for i, (test_label, test_p) in enumerate(zip(test_labels, test_pred)):
            if test_label == 1:
                if test_label == test_p:
                    selected_prediction_type_dict["TruePositive"].append(str(i))
                else:
                    selected_prediction_type_dict["FalseNegative"].append(str(i))
            else:
                if test_label == test_p:
                    selected_prediction_type_dict["TrueNegative"].append(str(i))
                else:
                    selected_prediction_type_dict["FalsePositive"].append(str(i))
        self.detailed_country_data[country] = {
            "Metadata": metadata_dict,
            "Details": selected_prediction_type_dict,
        }

    def get_country_data(self, country):
        self.calculate_details_for_country(country=country)
        return self.detailed_country_data[country]

    def get_global_data(self, country):
        return {
            "TopWords": self.global_data[country][:100],
            "BottomWords": self.global_data[country][-100:][::-1],
        }

    def get_tender_data(self, country, tender_id):
        (
            clf,
            vectorizer,
            (test_features, test_pred),
            test_metrics,
            (test_original_texts, test_texts, test_labels),
        ) = self.country_model_data[country]

        tender_index = int(tender_id)

        word_scores = test_features[tender_index].multiply(clf.coef_[0]).tocsr()
        scored_words = []
        total_score = test_features[tender_index].dot(clf.coef_[0])
        word_score = {}
        lemma_original = {}
        for i, (original_word, lemma_word) in enumerate(
            zip(
                test_original_texts[tender_index].split(" "),
                test_texts[tender_index].split(" "),
            )
        ):
            score = 0.0
            if lemma_word in vectorizer.vocabulary_:
                word_index = vectorizer.vocabulary_[lemma_word]
                score = word_scores[0, word_index].item()

            if lemma_word not in word_score:
                word_score[lemma_word] = 0
            if lemma_word not in lemma_original:
                lemma_original[lemma_word] = []
            word_score[lemma_word] = score
            lemma_original[lemma_word].append(original_word)

            scored_words.append([original_word, score])

        fig, ax = plt.subplots()
        word_score = dict(sorted(word_score.items(), key=lambda k: k[1]))
        vis_words = []
        current_sum = 0
        bias = np.array(clf.intercept_)
        ax.axvline(x=-bias[0], color="red", label="decision boundary")
        ax.text(
            -bias[0] + 0.05, 5, "decision boundary", rotation=90, color="r", va="center"
        )
        ax.axvline(x=0, color="black", label="zero", linestyle="dashed")

        top_keys = list(word_score.keys())[-5:]
        top_keys.reverse()
        for i, lemma_word in enumerate(top_keys):
            # if word_score[lemma_word] == 0:
            #    continue
            ax.barh(i, current_sum + word_score[lemma_word], align="center", color="r")
            ax.barh(i, current_sum, align="center", color="white")
            current_sum += word_score[lemma_word]
            vis_words.append(lemma_original[lemma_word][0].lower())
            ax.text(
                current_sum + 0.1,
                i,
                str(round(word_score[lemma_word], 2)),
                color="r",
                va="center",
            )

        bot_keys = list(word_score.keys())[:5]
        # bot_keys.reverse()
        for i, lemma_word in enumerate(bot_keys):
            # if word_score[lemma_word] == 0:
            #    continue
            if current_sum > 0:
                zero_to_pos = current_sum + word_score[lemma_word]
                ax.barh(i + 5, current_sum, align="center", color="b")
                if zero_to_pos > 0:
                    ax.barh(i + 5, zero_to_pos, align="center", color="white")
                else:
                    ax.barh(i + 5, current_sum, align="center", color="b")
                    ax.barh(i + 5, zero_to_pos, align="center", color="b")
            else:
                ax.barh(
                    i + 5,
                    current_sum + word_score[lemma_word],
                    align="center",
                    color="b",
                )
                ax.barh(i + 5, current_sum, align="center", color="white")
            ax.text(
                current_sum + 0.1,
                i + 5,
                str(round(word_score[lemma_word], 2)),
                color="blue",
                va="center",
            )
            current_sum += word_score[lemma_word]
            vis_words.append(lemma_original[lemma_word][0].lower())
        ax.set_yticks(np.linspace(0, 10, 10), labels=vis_words)
        ax.invert_yaxis()
        fig.tight_layout()

        filename = uuid.uuid4()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        byte_image = buf.read().hex()

        return {"WordScores": scored_words, "Plot": byte_image}
