import logging
import re
import sys
from collections import Counter, defaultdict
import textwrap

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import spacy
import json
from IPython.display import display, Markdown
from termcolor import colored
from wordfreq import zipf_frequency

from analytics_lib.nlp_texts.for_dict_processing import get_all_sence_df, verbs_df_to_internal_pred_verbs, verbs_df_to_external_pred_verbs
from analytics_lib.nlp_texts.psychotype import *

w = textwrap.TextWrapper(width = 180, replace_whitespace=False)

class TextProcessor:
    def __init__(self, m, nlp_core, morpholog, fastTextSocialNetworkModel, nlp_spacy, stemmer, elmo_model, df_sense, verbs_df):
        self.m = m
        self.nlp_core = nlp_core
        self.morpholog = morpholog
        self.fastTextSocialNetworkModel = fastTextSocialNetworkModel
        self.nlp_spacy = nlp_spacy
        self.stemmer = stemmer
        self.elmo_model = elmo_model
        self.df_sense = df_sense
        self.verbs_df = verbs_df


    @staticmethod
    def split_text(text, num_parts=3):
        length = len(text)
        shift = int(length / num_parts)
        parts = []
        k = 0
        while k + shift < length:
            j = k + shift
            if text[k + shift] == " ":
                parts.append(text[k : k + shift])
            else:
                g = text[k + shift : -1].find(" ")
                if g != -1:
                    j = g + k + shift
                    parts.append(text[k:j])
                else:
                    break
            k = j
        parts.append(text[k:-1])
        return parts

    @staticmethod
    def is_in_any(x, params_list):
        res = False
        for params in params_list:
            res |= x in params

        return res

    @staticmethod
    def double_is_in(double_tuple, params_list):
        for one_list in params_list:
            if "прич" in one_list:
                res = False
                continue
            res = True
            for word in double_tuple:
                if word not in one_list:
                    res = False
            if res:
                return res
        return res

    def morph_count(self, word):
        morpholog = self.morpholog
        return len(morpholog.tokenize(word)[0])

    @staticmethod
    def avg10_freq_of_rare_words(global_word_frequency):
        list_freq = list(global_word_frequency.values())
        list_freq.sort()
        return np.mean(list_freq[:10])

    @staticmethod
    def indicator_subordinate(processed_text_sentence):
        sentence = processed_text_sentence
        ind = 0
        for token in sentence.tokens:
            if not token.words:
                continue
            if token.words[0].deprel == "nsubj" and token.words[0].text.lower() in\
                                                    ["который", "которая", "которое", "которые", "что"]:
                ind = 1
            elif token.words[0].deprel == "mark":
                ind = 1
        return ind

    @staticmethod
    def split_sentence(sentence):
        list_of_res_sents = []
        list_temp_sents = sentence.split('\n')
        for element in list_temp_sents:
            if len(element) > 4:
                list_of_res_sents.append(element)
        return list_of_res_sents

    @staticmethod
    def is_spam(sentence):
        flag = False
        spam_token = ""
        list_tokens = sentence.split(" ")
        for token in list_tokens:
            if len(token) > 40:
                flag = True
                spam_token = token
        return flag, spam_token

    def list_of_sentences(self, processed_text):
        list_of_stanza_sentences = []
        list_res_of_sents = []
        for sentence in processed_text.sentences:
            list_of_stanza_sentences.append(sentence.text)
        for sentence in list_of_stanza_sentences:
            temp_list = self.split_sentence(sentence)
            for sent in temp_list:
                list_res_of_sents.append(sent)
        last_sen = list_res_of_sents[-1]
        clean_last_sen = ""
        spam_ind = self.is_spam(last_sen)
        if spam_ind[0]:
            clean_last_sen = last_sen[:last_sen.index(spam_ind[1])]
            if len(clean_last_sen) > 4:
                list_res_of_sents[-1] = clean_last_sen
            else:
                list_res_of_sents = list_res_of_sents[:-1]
            
        return list_res_of_sents

    @staticmethod
    def list_of_sentences_len(list_of_sentences, list_of_words):
        list_of_lens = []
        for sentence in list_of_sentences:
            count = 0
            sent_without_punct = re.sub(r'[^\w\s]', '', sentence)
            list_of_tokens = sent_without_punct.split()
            for token in list_of_tokens:
                if token in list_of_words:
                    count += 1
            list_of_lens.append(count)
        return list_of_lens

    def process_processed_text(self, processed_text):
        features_counter = {
            "text_len": 0,
            "sentences_len": [],
            "mean_words_len": 0,
            "sentences_mean_words_len": [],
            "std_words_len": 0,
            "sentences_std_words_len": [],
            "text_tags": defaultdict(int),
            "sentences_tags": [],
            "text_punctuation": defaultdict(int),
            "sentences_punctuation": [],
            "text_converbs": 0,
            "sentences_converbs": [],
            "text_conditionals": 0,
            "sentences_conditionals": [],
            "text_moods": defaultdict(int),
            "sentences_moods": [],
            "text_verb_voices": defaultdict(int),
            "sentences_verb_voices": [],
            "text_persons": defaultdict(int),
            "sentences_persons": [],
            "sentences_adj_for_noun": [],
            "sentences_verb_for_verb": [],
            "text_word_frequency": defaultdict(int),
            "global_word_frequency": defaultdict(int),
            "sentences_word_frequency": [],
            "avg10_freq_of_rare_words": 0,
            "O_counter": 0,
            "U_counter": 0,
            "A_counter": 0,
            "vowel_counter": 0,
            "consonant_counter": 0,
            "name_counter": [],
            "morph_counter": 0,
            "letter_counter": 0,
            "neg_parts_counter": 0,
            "count_long_low_freq_words": 0,
            "count_short_high_freq_words": 0,
            "count_subordinate": 0, 
            "number_of_sentences": 0,
            "list_of_sentences_len": [],
            "list_important_words": [],
            "list_unique_words": [],
            "list_repeated_words": [],
            "input_word_frequency": defaultdict(int),
            "list_pronouns": [],
            "list_parts": [],
            "list_num": [],
            "list_intj": [],
            "list_prep": [], 
            "list_conj": [],
            
        }

        # list_of_sents = self.list_of_sentences(processed_text)
        # features_counter["number_of_sentences"] = len(list_of_sents)
        # features_counter["list_of_sentences_len"] = self.list_of_sentences_len(list_of_sents, list_of_words)

        text_words_len = []
        input_word_frequency = defaultdict(int)
        global_word_frequency = defaultdict(int)

        list_of_words = []
        list_important_words = [] # для водности
        text_lemma_frequency = defaultdict(int)
        list_unique_words = [] # для вариабельности
        list_repeated_words = [] # для повторов
        list_pronouns = []
        list_parts = []
        list_num = []
        list_intj = []
        list_prep = []
        list_conj = []

        for sentence in processed_text.sentences:
            sentence_len_counter = 0
            sentence_name_counter = 0
            sentence_words_len_counter = []
            sentence_tags_counter = defaultdict(int)
            sentence_punctuation_counter = defaultdict(int)
            sentence_converbs_counter = 0
            sentence_conditionals_counter = 0
            sentence_moods_counter = defaultdict(int)
            sentence_verb_voices_counter = defaultdict(int)
            sentence_person_counter = defaultdict(int)
            sentence_adj_for_noun_counter = defaultdict(int)
            sentence_verb_for_verb = 0
            sentence_word_frequency = defaultdict(int)

            #сложноподчиненные предложения
            features_counter["count_subordinate"] += self.indicator_subordinate(sentence)
            

            for token in sentence.tokens:
                if not token.words:
                    continue
                    
                token = token.words[0]

                if len(token.text) > 40:
                    continue
                    
                if token.upos == 'ADP':
                    list_prep.append(token.text)
                elif token.upos == 'CCONJ':
                    list_conj.append(token.text)
                    
                if token.upos == 'NUM':
                    list_num.append(token.text)
                elif token.upos == 'INTJ':
                    list_intj.append(token.text)
                    
                if token.upos not in ["PUNCT", "ADP", "CCONJ", "SCONJ", "SYM", "NUM",\
                                      "PART", "AUX", "PRON", "INTJ"]:
                    list_important_words.append(token.lemma.lower())

                if token.upos == "PART" and token.text in ["Не", "не", "НЕ", "нЕ"]:
                    features_counter["neg_parts_counter"] += 1

                if token.upos not in ["PUNCT", "ADP", "CCONJ", "SCONJ"]:  # вот тут перестаем считать предлоги
                    text_lemma_frequency[token.lemma.lower()] += 1 # для вариабельности
                    features_counter["text_len"] += 1
                    sentence_len_counter += 1
                    list_of_words.append(token.text)

                    features_counter["morph_counter"] += self.morph_count(token.text)

                    input_word_frequency[token.text.lower()] += 1
                    features_counter["text_word_frequency"][token.text.lower()] += 1
                    sentence_word_frequency[token.text.lower()] += 1

                    token_len = 0
                    word_len = 0
                    for letter in token.text.lower():
                        #token_len += 1
                        if letter == "о":
                            features_counter["O_counter"] += 1
                        elif letter == "и":
                            features_counter["U_counter"] += 1
                        elif letter == "а":
                            features_counter["A_counter"] += 1

                        if letter in ["а", "у", "о", "ы", "и", "э", "я", "ю", "ё", "е"] + ["a", "e", "i", "o", "u"]:
                            features_counter["vowel_counter"] += 1
                            word_len += 1
                        elif letter in ["б", "в", "г", "д", "ж", "з", "й", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"] + ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"]:
                            features_counter["consonant_counter"] += 1
                            word_len += 1
                        elif letter in ["ъ", "ь"]:
                            word_len += 1


                    sentence_words_len_counter.append(word_len)
                    text_words_len.append(word_len)

                sentence_tags_counter[token.upos] += 1
                features_counter["text_tags"][token.upos] += 1

                try:
                    mood = token.feats.split("Mood=")[1].split("|")[0]

                    sentence_moods_counter[mood] += 1
                    features_counter["text_moods"][mood] += 1
                except:
                    pass
                
                if token.upos == 'PRON':
                    list_pronouns.append(token.text)
                elif token.upos == 'PART':
                    list_parts.append(token.text)

                if token.upos == "PRON":
                    try:
                        pron_person = token.feats.split("Person=")[1].split("|")[0]

                        sentence_person_counter[pron_person] += 1
                        features_counter["text_persons"][pron_person] += 1
                    except:
                        continue

                elif token.upos == "PUNCT":
                    sentence_punctuation_counter[token.text] += 1
                    features_counter["text_punctuation"][token.text] += 1
                elif token.upos in ["VERB", "AUX"]:
                    if sentence.tokens[token.head - 1].words[0].upos == "VERB":
                        sentence_verb_for_verb += 1
                    if token.text.lower() == "бы":
                        sentence_conditionals_counter += 1
                        features_counter["text_conditionals"] += 1

                    try:
                        verb_form = token.feats.split("VerbForm=")[1].split("|")[0]
                    except:
                        continue

                    if verb_form == "Conv":
                        sentence_converbs_counter += 1
                        features_counter["text_converbs"] += 1
                    else:
                        try:
                            verb_voice = token.feats.split("Voice=")[1].split("|")[0]

                            sentence_verb_voices_counter[verb_voice] += 1
                            features_counter["text_verb_voices"][verb_voice] += 1
                        except:
                            continue
                elif token.upos == "ADJ":
                    try:
                        sentence_adj_for_noun_counter[str(token.head)] += 1
                    except:
                        pass

                elif token.upos == "PROPN":
                    sentence_name_counter += 1
                    
            if sentence_len_counter > 0:
                features_counter["sentences_len"].append(sentence_len_counter)
                features_counter["sentences_mean_words_len"].append(np.around(np.nanmean(np.array([int(value) for value in sentence_words_len_counter]))))
                features_counter["sentences_std_words_len"].append(np.nanstd(np.array(sentence_words_len_counter)))
                features_counter["sentences_tags"].append(sentence_tags_counter)
                features_counter["sentences_punctuation"].append(sentence_punctuation_counter)
                features_counter["sentences_converbs"].append(sentence_converbs_counter)
                features_counter["sentences_conditionals"].append(sentence_conditionals_counter)
                features_counter["sentences_moods"].append(sentence_moods_counter)
                features_counter["sentences_verb_voices"].append(sentence_verb_voices_counter)
                features_counter["sentences_persons"].append(sentence_person_counter)
                features_counter["sentences_adj_for_noun"].append(np.nanmean(np.array([int(value) for value in sentence_adj_for_noun_counter.values()])))
                features_counter["sentences_verb_for_verb"].append(sentence_verb_for_verb)
                features_counter["sentences_word_frequency"].append(sentence_word_frequency)
                features_counter["name_counter"].append(sentence_name_counter)

        features_counter["mean_words_len"] = np.nanmean(np.array(text_words_len))
        features_counter["std_words_len"] = np.nanstd(np.array(text_words_len))
        features_counter["letter_counter"] = sum(text_words_len)

        for word in input_word_frequency.keys():
            global_word_frequency[word] = zipf_frequency(word, "ru")
        features_counter["global_word_frequency"] = global_word_frequency

        for word in input_word_frequency.keys():
            if len(word) > 9.6 and global_word_frequency[word] < 4.037:
                features_counter["count_long_low_freq_words"] += 1
            elif len(word) < 9.6 and global_word_frequency[word] > 4.037:
                features_counter["count_short_high_freq_words"] += 1
                
        features_counter["input_word_frequency"] = input_word_frequency
        features_counter["list_pronouns"] = list_pronouns
        features_counter["list_parts"] = list_parts
        


        features_counter["avg10_freq_of_rare_words"] = TextProcessor.avg10_freq_of_rare_words(global_word_frequency)

        list_of_sents = self.list_of_sentences(processed_text)
        features_counter["number_of_sentences"] = len(list_of_sents)
        features_counter["list_of_sentences_len"] = self.list_of_sentences_len(list_of_sents, list_of_words)
        
        # для новых признаков
        features_counter["list_important_words"] = list_important_words
        features_counter["list_unique_words"] = [word for word in text_lemma_frequency.keys()]
        k = 2
        features_counter["list_repeated_words"] = [word for word in text_lemma_frequency.keys() if text_lemma_frequency[word] >= k]
        
        features_counter["list_num"] = list_num
        features_counter["list_intj"] = list_intj
        features_counter["list_prep"] = list_prep
        features_counter["list_conj"] = list_conj

        return features_counter

    @staticmethod
    def tree_count_of_f(root, f):
        if not list(root.children):
            if f(root):
                return 1
            else:
                return 0
        else:
            if f(root):
                return 1 + sum(TextProcessor.tree_count_of_f(x, f) for x in root.children)
            else:
                return sum(TextProcessor.tree_count_of_f(x, f) for x in root.children)

    @staticmethod
    def avg_tree_count_of_f(doc, f):
        roots = [sent.root for sent in doc.sents]
        return np.mean(
            [
                TextProcessor.tree_count_of_f(root, f) / len([child for child in root.children])
                if len([child for child in root.children]) != 0
                else 0
                for root in roots
            ]
        )

    @staticmethod
    def tree_height(root):
        if not list(root.children):
            return 1
        else:
            return 1 + max(TextProcessor.tree_height(x) for x in root.children)

    @staticmethod
    def get_average_heights(doc):
        roots = [sent.root for sent in doc.sents]
        return np.mean([TextProcessor.tree_height(root) for root in roots])

    def get_syntax_branches_stats(self, text):
        nlp = self.nlp_spacy
        doc = nlp(text)

        number_branches = 0

        total_nodes_with_mt_2_children = 0
        total_children_of_them = 0
        mean_count_of_syntax_branches_by_one_word = 0

        for token in doc:
            list_of_children = [child for child in token.children]
            number_children_of_node = len(list_of_children)
            if token.dep_ == "ROOT":
                number_branches += number_children_of_node
            if number_children_of_node > 1:
                total_children_of_them += number_children_of_node
                total_nodes_with_mt_2_children += 1

        count_of_syntax_branches = number_branches
        if total_nodes_with_mt_2_children > 0:
            mean_count_of_syntax_branches_by_one_word = total_children_of_them / total_nodes_with_mt_2_children

        avg_depth_of_sent = TextProcessor.get_average_heights(doc)
        avg_tree_count_amod = TextProcessor.avg_tree_count_of_f(doc, lambda x: x.dep_ == "amod")
        avg_tree_count_acl = TextProcessor.avg_tree_count_of_f(doc, lambda x: x.dep_ == "acl")
        avg_tree_count_advcl = TextProcessor.avg_tree_count_of_f(doc, lambda x: x.dep_ == "advcl")

        return (
            count_of_syntax_branches,
            mean_count_of_syntax_branches_by_one_word,
            avg_depth_of_sent,
            avg_tree_count_amod,
            avg_tree_count_acl,
            avg_tree_count_advcl,
        )

    def get_stat(self, text):

        text = text.replace("\n", " ")

        nlp_core = self.nlp_core
        m = self.m

        (
            count_of_syntax_branches,
            mean_count_of_syntax_branches_by_one_word,
            avg_depth_of_sent,
            avg_tree_count_amod,
            avg_tree_count_acl,
            avg_tree_count_advcl,
        ) = self.get_syntax_branches_stats(text)
        syntax_branches_counter = Counter()

        syntax_branches_counter["Количество синтаксических веток"] = count_of_syntax_branches
        syntax_branches_counter["Среднее число веток на одно слово"] = mean_count_of_syntax_branches_by_one_word
        syntax_branches_counter["Средняя глубина ветки"] = avg_depth_of_sent
        syntax_branches_counter["Среднее число прилагательных в ветке"] = avg_tree_count_amod
        syntax_branches_counter["Среднее число причастий в ветке"] = avg_tree_count_acl
        syntax_branches_counter["Среднее число деепричастий в ветке"] = avg_tree_count_advcl

        counter = Counter()
        verb_type_counter = Counter()
        adj_counter = Counter()
        adj_degree_counter = Counter()

        adv_counter = Counter()
        participle_counter = Counter()
        gerunds_counter = Counter()

        comparison_degrees = Counter()
        naklonenie = Counter()
        analysis = m.analyze(text)

        counter["Причастие"] = 0
        counter["Деепричастие"] = 0
        counter["Глагол"] = 0
        counter["Прилагательное"] = 0
        counter["Наречие"] = 0
        counter["Существительные"] = 0
        naklonenie["Изъявительные"] = 0
        naklonenie["Повелительные"] = 0
        
        # для новых признаков
        counter["Собственные"] = 0
        counter["Нарицательные"] = 0

        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V":
                    if "изъяв" in part or TextProcessor.is_in_any("изъяв", params_list):
                        naklonenie["Изъявительные"] += 1
                    elif "пов" in part or TextProcessor.is_in_any("пов", params_list):
                        naklonenie["Повелительные"] += 1

                    if "прич" in part or TextProcessor.is_in_any("прич", params_list):
                        counter["Причастие"] += 1
                        if "страд" in part or TextProcessor.is_in_any("страд", params_list):
                            participle_counter["Страдательные"] += 1
                        elif "действ" in part or TextProcessor.is_in_any("действ", params_list):
                            participle_counter["Действительные"] += 1
                        continue

                    elif "деепр" in part or TextProcessor.is_in_any("деепр", params_list):
                        counter["Деепричастие"] += 1
                        if "сов" in part or TextProcessor.is_in_any("сов", params_list):
                            gerunds_counter["Совершенный вид"] += 1
                        elif "несов" in part or TextProcessor.is_in_any("несов", params_list):
                            gerunds_counter["Несовершенный вид"] += 1
                        continue

                    counter["Глагол"] += 1
                    if "сов" in part or TextProcessor.is_in_any("сов", params_list):
                        verb_type_counter["Совершенный вид"] += 1
                    elif "несов" in part or TextProcessor.is_in_any("несов", params_list):
                        verb_type_counter["Несовершенный вид"] += 1
                    else:
                        logging.info("Проблема с глаголами", token_data["analysis"])

                elif speech_part == "A":
                    counter["Прилагательное"] += 1
                    if "притяж" in part or TextProcessor.is_in_any("притяж", params_list):
                        adj_counter["Притяжательные"] += 1
                    elif "прев" in part or TextProcessor.is_in_any("прев", params_list):
                        comparison_degrees["Превосходная"] += 1
                        adj_counter["Качественные"] += 1
                    elif "срав" in part or TextProcessor.is_in_any("срав", params_list):
                        comparison_degrees["Сравнительная"] += 1
                        adj_counter["Качественные"] += 1
                    else:
                        logging.info("Проблема с прилагательными", token_data["analysis"])

                elif speech_part == "ADV":
                    counter["Наречие"] += 1
                    if "прев" in part or TextProcessor.is_in_any("прев", params_list):
                        comparison_degrees["Превосходная"] += 1
                    elif "срав" in part or TextProcessor.is_in_any("срав", params_list):
                        comparison_degrees["Сравнительная"] += 1

                elif speech_part == "S":
                    counter["Существительные"] += 1
                    if "имя" in part or TextProcessor.is_in_any("имя", params_list):
                        counter["Собственные"] += 1
                    else:
                        counter["Нарицательные"] += 1

        processed_text = nlp_core(text)
        features_counter = self.process_processed_text(processed_text)

        return (
            counter,
            verb_type_counter,
            adj_counter,
            adv_counter,
            participle_counter,
            gerunds_counter,
            comparison_degrees,
            naklonenie,
            features_counter,
            syntax_branches_counter,
        )

    def get_sentiments(self, text):
        parts = TextProcessor.split_text(text)
        model1 = self.fastTextSocialNetworkModel
        sentiment_by_text_parts = model1.predict(sentences=parts, k=5)
        full_text_sentiment = model1.predict(sentences=[text], k=5)
        sentiment_by_text_parts.append(full_text_sentiment[0])
        return sentiment_by_text_parts

    def color_verbs(self, text):
        m = self.m
        analysis = m.analyze(text)
        counter = Counter()
        counter["Будущее"] = 0
        counter["Настоящее"] = 0
        counter["Прошедшее"] = 0
        counter["Глагол"] = 0
        counter["Абсолютное"] = 0
        i = 0
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if (
                    (speech_part == "V")
                    and not (TextProcessor.is_in_any("прич", params_list))
                    and not (TextProcessor.is_in_any("деепр", params_list))
                ):
                    if TextProcessor.is_in_any("прош", params_list) or "прош" in part:
                        text = text.replace(word, colored(word, "red"))
                        counter["Прошедшее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("непрош", params_list) and ("сов" in part or TextProcessor.is_in_any("сов", params_list)):
                        text = text.replace(word, colored(word, "blue"))
                        counter["Будущее"] += 1
                        counter["Глагол"] += 1
                    elif (
                        TextProcessor.is_in_any("непрош", params_list)
                        and not ("сов" in part or TextProcessor.is_in_any("сов", params_list))
                        and not ("несов" in part or TextProcessor.is_in_any("несов", params_list))
                    ):
                        text = text.replace(word, colored(word, "blue"))
                        counter["Будущее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("наст", params_list) or TextProcessor.is_in_any("непрош", params_list):
                        text = text.replace(word, colored(word, "green"))
                        counter["Настоящее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("инф", params_list):
                        text = text.replace(word, colored(word, "white"))
                        counter["Абсолютное"] += 1
                        counter["Глагол"] += 1
                    else:
                        i += 1
                        counter["Глагол"] += 1

        s = f"""
        * Доля глаголов {colored("будущего", "blue")} времени, %: {100 * counter["Будущее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1):.2f}

        * Доля глаголов {colored("прошедшего", "red")} времени, %: {100 * counter["Прошедшее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1):.2f}

        * Доля глаголов {colored("настоящего", "green")} времени, %: {100 * counter["Настоящее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1):.2f}

        * Доля глаголов {colored("абсолютного", "white")} времени, %: {100 * counter["Абсолютное"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1):.2f}
        """
        print(text)
        print(s)

        return text

    @staticmethod
    def contain_force(word):
        list_of_force = ["долж", "нужд"]
        for force in list_of_force:
            if force in word.lower():
                return True
        return False

    @staticmethod
    def remove_dashes(text):
        text = text.replace(" - ", " ").replace(" — ", " ").replace("- ", " ").replace("— ", " ").replace(" -", " ").replace(" —", " ")
        return text



    def lists_of_past_time_verbs(self, text):
        list_of_multi_past_time_verbs = []
        list_of_single_past_time_verbs_mf_g = []
        list_of_single_past_time_verbs_n_g = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V":
                    if TextProcessor.double_is_in(("прош", "мн"), params_list):
                        list_of_multi_past_time_verbs.append(word)
                    elif TextProcessor.double_is_in(("прош", "ед"), params_list) and TextProcessor.is_in_any("сред", params_list):
                        list_of_single_past_time_verbs_n_g.append(word)
                    elif TextProcessor.double_is_in(("прош", "ед"), params_list):
                        list_of_single_past_time_verbs_mf_g.append(word)

        return list_of_single_past_time_verbs_mf_g, list_of_multi_past_time_verbs, list_of_single_past_time_verbs_n_g
    
    def list_of_agent_verbs(self, text):
        list_of_agent_verbs = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V" and not word.lower().startswith("есть") and not word.lower().startswith("нужд"):
                    if TextProcessor.double_is_in(("1-л", "ед"), params_list):
                        list_of_agent_verbs.append(word)
        return list_of_agent_verbs

    def list_of_single_past_time_verbs(self, text):
        list_of_single_past_time_verbs = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V":
                    if TextProcessor.double_is_in(("прош", "ед"), params_list) and not TextProcessor.is_in_any("сред", params_list) and not word.lower().startswith("нужд"):
                        list_of_single_past_time_verbs.append(word)
        return list_of_single_past_time_verbs

    def list_of_pers_patient_verbs(self, text):
        list_of_pers_patient_verbs = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]
                

                if speech_part == "V":
                    if TextProcessor.double_is_in(("1-л", "ед"), params_list) and (word.lower().startswith("нужд") or word.lower().startswith("есть")):
                        list_of_pers_patient_verbs.append(word)
                    elif TextProcessor.double_is_in(("1-л", "мн"), params_list):
                        list_of_pers_patient_verbs.append(word)
                    elif TextProcessor.is_in_any("2-л", params_list):
                        list_of_pers_patient_verbs.append(word)
                    elif TextProcessor.is_in_any("3-л", params_list):
                        list_of_pers_patient_verbs.append(word)
                    
                        
                    
        return list_of_pers_patient_verbs


    def list_of_inf_verbs_from_text(self, text):
        list_of_inf = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V" and TextProcessor.is_in_any("инф", params_list):
                    list_of_inf.append(word)
        return list_of_inf


    def list_of_agence(self, text):
        dict_with_norm = self.dict_verbs_norm(text)
        verbs_from_text = list(dict_with_norm.keys())
        agent_verbs = self.list_of_agent_verbs(text)
        single_past_time_verbs = self.list_of_single_past_time_verbs(text)
        count = len(verbs_from_text)
        if count == 0:
            count += 1
        #text = TextProcessor.remove_dashes(text)
        doc = self.nlp_spacy(text)
        list_of_pattern = []
        list_of_pattern_i = []
        for token in doc:
            if token.text in agent_verbs and not TextProcessor.contain_force(token.head.text):
                list_of_colloc = []
                list_of_index = []
                list_of_colloc.append(token.text)
                list_of_index.append(token.i)
                for child in token.children:
                    if child.dep_ == "nsubj" and child.text.lower() == "я":
                        list_of_colloc.append(child.text)
                        list_of_index.append(child.i)
                list_of_pattern.append(tuple(list_of_colloc))
                list_of_pattern_i.append(tuple(list_of_index))
            elif token.text in single_past_time_verbs and not TextProcessor.contain_force(token.head.text):
                list_of_colloc = []
                list_of_index = []
                list_of_colloc.append(token.text)
                list_of_index.append(token.i)
                there_exist_agent_nsubj = False
                there_exist_nsubj = False
                for child in token.children:
                    if child.dep_ == "nsubj" and child.text.lower() == "я":
                        there_exist_agent_nsubj = True
                        list_of_colloc.append(child.text)
                        list_of_index.append(child.i)
                    elif child.dep_ == "nsubj":
                        there_exist_nsubj = True
                if not (there_exist_nsubj or there_exist_agent_nsubj):
                    for child in token.head.children:
                        if child.dep_ == "nsubj"and child.text.lower() == "я":
                            there_exist_agent_nsubj = True
                            list_of_colloc.append(child.text)
                            list_of_index.append(child.i)
                        elif child.dep_ == "nsubj":
                            there_exist_nsubj = True
                        
                if there_exist_agent_nsubj:       
                    list_of_pattern.append(tuple(list_of_colloc))
                    list_of_pattern_i.append(tuple(list_of_index))
                elif not there_exist_nsubj:
                    list_of_pattern.append(tuple(list_of_colloc))
                    list_of_pattern_i.append(tuple(list_of_index))

        return list_of_pattern, list_of_pattern_i, count


    def list_of_not_agence(self, text):
        dict_with_norm = self.dict_verbs_norm(text)
        verbs_from_text = list(dict_with_norm.keys())
        count = len(verbs_from_text)
        if count == 0:
            count += 1
        #text = TextProcess
        list_of_single_past_time_verbs_mf_g, list_of_multi_past_time_verbs, list_of_single_past_time_verbs_n_g = self.lists_of_past_time_verbs(text)
        list_of_personal_patient_verbs = self.list_of_pers_patient_verbs(text)
        list_of_inf = self.list_of_inf_verbs_from_text(text)
        doc = self.nlp_spacy(text)
        list_of_pattern = []
        list_of_pattern_i = []
        for token in doc:
            if token.text in list_of_personal_patient_verbs or token.text in list_of_inf or token.text in list_of_multi_past_time_verbs or token.text in list_of_single_past_time_verbs_n_g:
                list_of_colloc = []
                list_of_index = []
                list_of_colloc.append(token.text)
                list_of_index.append(token.i)
                list_of_pattern.append(tuple(list_of_colloc))
                list_of_pattern_i.append(tuple(list_of_index))
            elif token.text in list_of_single_past_time_verbs_mf_g:
                list_of_colloc = []
                list_of_index = []
                list_of_colloc.append(token.text)
                list_of_index.append(token.i)
                there_exist_nsubj = False
                for child in token.children:
                    if child.dep_ == "nsubj" and child.text.lower() != "я":
                        there_exist_nsubj = True
                for child in token.head.children:
                    if child.dep_ == "nsubj"and child.text.lower() != "я":
                        there_exist_nsubj = True
                if there_exist_nsubj:
                    list_of_pattern.append(tuple(list_of_colloc))
                    list_of_pattern_i.append(tuple(list_of_index))

        return list_of_pattern, list_of_pattern_i, count


    def dict_of_morphology_features(self, text):
        orig_text = text
        m = self.m
        analysis = m.analyze(text)
        counter = Counter()
        counter["Будущее"] = 0
        counter["Настоящее"] = 0
        counter["Прошедшее"] = 0
        counter["Глагол"] = 0
        counter["Абсолютное"] = 0
        i = 0
        list_of_inf = [] #список глаголов в инфинитиве
        list_of_present = []
        list_of_past = []
        list_of_future = []
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if (
                    (speech_part == "V")
                    and not (TextProcessor.is_in_any("прич", params_list))
                    and not (TextProcessor.is_in_any("деепр", params_list))
                ):
                    if TextProcessor.is_in_any("прош", params_list) or "прош" in part:
                        text = text.replace(word, colored(word, "red"))
                        list_of_past.append(word)
                        counter["Прошедшее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("непрош", params_list) and ("сов" in part or TextProcessor.is_in_any("сов", params_list)):
                        text = text.replace(word, colored(word, "blue"))
                        list_of_future.append(word)
                        counter["Будущее"] += 1
                        counter["Глагол"] += 1
                    elif (
                        TextProcessor.is_in_any("непрош", params_list)
                        and not ("сов" in part or TextProcessor.is_in_any("сов", params_list))
                        and not ("несов" in part or TextProcessor.is_in_any("несов", params_list))
                    ):
                        text = text.replace(word, colored(word, "blue"))
                        list_of_future.append(word)
                        counter["Будущее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("наст", params_list) or TextProcessor.is_in_any("непрош", params_list):
                        text = text.replace(word, colored(word, "green"))
                        list_of_present.append(word)
                        counter["Настоящее"] += 1
                        counter["Глагол"] += 1
                    elif TextProcessor.is_in_any("инф", params_list):
                        text = text.replace(word, colored(word, "white"))
                        list_of_inf.append(word)
                        counter["Абсолютное"] += 1
                        counter["Глагол"] += 1

                    else:
                        i += 1
                        counter["Глагол"] += 1

        s = f"""
        * Доля глаголов {colored("будущего", "blue")} времени, %: {100 * counter["Будущее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1)}
         
        * Доля глаголов {colored("прошедшего", "red")} времени, %: {100 * counter["Прошедшее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1)}
          
        * Доля глаголов {colored("настоящего", "green")} времени, %: {100 * counter["Настоящее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1)}

        * Доля глаголов {colored("абсолютного", "green")} времени, %: {100 * counter["Абсолютное"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1)}
        """

        dict_res = {"Текст": orig_text,
        "Прошедшее, %" : 100 * counter["Прошедшее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1), 
        "Настоящее, %": 100 * counter["Настоящее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1),
         "Будущее, %": 100 * counter["Будущее"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1),
          "Абсолютное, %": 100 * counter["Абсолютное"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1),
          "Прошедшее, слова": list_of_past,
          "Настоящее, слова": list_of_present,
          "Будущее, слова": list_of_future,
          "Абсолютное, слова": list_of_inf}

        return dict_res

    def df_of_morphology_features(self, text):
        dict_res = self.dict_of_morphology_features(text)
        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = np.round(pd.DataFrame.from_dict(dict_res), 1)
        return df


    def dict_of_semantic_role_features(self, text):
        list_of_agences = self.list_of_agence(text)
        list_of_not_agences = self.list_of_not_agence(text)
        list_of_internal_predicates = self.list_of_internal_predicat(text)
        list_of_external_predicates = self.list_of_external_predicat(text)

        dict_res = {"Текст": text,
        "Агенс" : round(len(list_of_agences[0]) / list_of_agences[2], 2), 
        "Пациенс": round(len(list_of_not_agences[0]) / list_of_not_agences[2], 2),
         "Внутренний предикат": round(len(list_of_internal_predicates[0]) / list_of_internal_predicates[2], 2),
          "Внешний предикат": round(len(list_of_external_predicates[0]) / list_of_external_predicates[2], 2), 
          "Агенс, конструкции": [list_of_agences[0]],
          "Пациенс, конструкции": [list_of_not_agences[0]],
          "Внутренний предикат, конструкции": [list_of_internal_predicates[0]],
          "Внешний предикат, конструкции": [list_of_external_predicates[0]],
          "Агенс, индексы": [list_of_agences[1]],
          "Пациенс, индексы": [list_of_not_agences[1]], 
          "Внутренний предикат, индексы": [list_of_internal_predicates[1]],
          "Внешний предикат, индексы": [list_of_external_predicates[1]]
          }

        return dict_res

    def df_of_semantic_role_features(self, text):
        dict_res = self.dict_of_semantic_role_features(text)
        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = np.round(pd.DataFrame.from_dict(dict_res), 1)
        return df


    @staticmethod
    def short_long_sent_count(list_of_sent_len):
        short_count = 0
        long_count = 0
        for length in list_of_sent_len:
            if length > 9:
                long_count += 1
            elif length < 9:
                short_count += 1
        return short_count, long_count


    @staticmethod
    def question_sent_count(list_of_dicts_with_sent_punct_count):
        res = 0
        for sent_dict in list_of_dicts_with_sent_punct_count:
            if sent_dict["?"] > 0:
                res += 1
        return res

    @staticmethod
    def exc_sent_count(list_of_dicts_with_sent_punct_count):
        res = 0
        for sent_dict in list_of_dicts_with_sent_punct_count:
            if sent_dict["!"] > 0:
                res += 1
        return res





    def dict_of_text_features(self, text):
 
        try:
            (
                counter,
                verb_type_counter,
                adj_counter,
                adv_counter,
                participle_counter,
                gerunds_counter,
                comparison_degrees,
                naklonenie,
                alpha_counter,
                syntax_branches_counter,
            ) = self.get_stat(text)

            sentiments = self.get_sentiments(text)
            naklonenie["Условные"] = sum(alpha_counter["sentences_conditionals"])

            neg_parts_count = alpha_counter["neg_parts_counter"]
            exc_count = self.exc_sent_count(alpha_counter["sentences_punctuation"])
            quest_count = self.question_sent_count(alpha_counter["sentences_punctuation"])
            list_of_sentences_len = alpha_counter["sentences_len"]
            short_count, long_count = self.short_long_sent_count(list_of_sentences_len)
            count_long_low_freq_words = alpha_counter["count_long_low_freq_words"]
            count_short_high_freq_words = alpha_counter["count_short_high_freq_words"]

            total1 = alpha_counter["text_len"]
            total = total1 if total1 != 0 else 1

            number_sentences1 = alpha_counter["number_of_sentences"]
            number_sentences = number_sentences1 if number_sentences1 != 0 else 1

            input_lemma_frequency = Counter(alpha_counter["list_important_words"])
            high_freq_word = get_key_with_max_value(input_lemma_frequency) \
                                if input_lemma_frequency else None
            classic_sickness = np.sqrt(Counter(alpha_counter["list_important_words"]).\
                                     get(high_freq_word)) if high_freq_word else 0
            sorted_dict = {k: v for k, v in sorted(input_lemma_frequency.items(),\
                                                    key=lambda item: item[1], reverse = True)}
            head_size = round(0.1 * total)
            df_head = TextProcessor.df_head_of_dict_freq(sorted_dict, head_size=head_size)
            df_kernel = df_head[df_head['count'] > 1]
            sum_top_freq = df_kernel['count'].sum()
            del df_kernel
            del df_head
            sum_top_freq = sum_top_freq if sum_top_freq else head_size

            letter_counter = alpha_counter["letter_counter"] if alpha_counter["letter_counter"] != 0 else 1

            dict_res = {
                "Текст": text,
                "Общее количество слов в тексте": total,
                "Доля предложений на одно слово": round(number_sentences / total, 2),
                "Доля коротких предложений": round(short_count / number_sentences, 2),
                "Доля длинных предложений": round(long_count / number_sentences, 2),
                "Доля длинных низкочастотных слов": round(count_long_low_freq_words / total, 2),
                "Доля коротких высокочастотных слов": round(count_short_high_freq_words / total, 2),
                "Средняя длина предложения": round(alpha_counter["text_len"] / number_sentences, 2),
                "Среднее число запятых в предложении": round(sum([x[","] for x in alpha_counter["sentences_punctuation"]]) / number_sentences, 2),
                "Доля сложноподчиненных предложений": round(alpha_counter["count_subordinate"] / number_sentences, 2),
                "Доля восклицательных предложений": round(exc_count / number_sentences, 2),
                "Доля вопросительных предложений": round(quest_count / number_sentences, 2),
                "Доля частиц НЕ": round(neg_parts_count / number_sentences, 2), 
                "Cредняя длинна слова в тексте": round(alpha_counter["mean_words_len"], 2),
                "Cредняя длинна слова по предложениям": alpha_counter["sentences_mean_words_len"],
                "Гласные, %": round(100 * alpha_counter["vowel_counter"] / letter_counter),
                'Гласная "И", %': round(100 * alpha_counter["U_counter"] / letter_counter),
                'Гласная "О", %': round(100 * alpha_counter["O_counter"] / letter_counter),
                'Гласная "А", %': round(100 * alpha_counter["A_counter"] / letter_counter),
                "Согласные, %": round(100 * alpha_counter["consonant_counter"] / letter_counter),
                "Существительные, %": round(100 * counter["Существительные"] / total, 2),
                "Глаголы, %": round(100 * counter["Глагол"] / total, 2),
                "Совершенный вид, %": round(100 * verb_type_counter["Совершенный вид"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1), 2),
                "Несовершенный вид, %": round(100 * verb_type_counter["Несовершенный вид"] / (counter["Глагол"] if counter["Глагол"] != 0 else 1), 2),
                "Прилагательные, %": round(100 * counter["Прилагательное"] / total, 2),
                "Степени сравнения, %": round(
                    100
                    * (comparison_degrees["Сравнительная"] + comparison_degrees["Превосходная"])
                    / (counter["Прилагательное"] if counter["Прилагательное"] != 0 else 1),
                    2,
                ),
                "Превосходная степень, %": round(
                    100
                    * (comparison_degrees["Превосходная"])
                    / (
                        comparison_degrees["Сравнительная"] + comparison_degrees["Превосходная"]
                        if (comparison_degrees["Сравнительная"] + comparison_degrees["Превосходная"]) != 0
                        else 1
                    ),
                    2,
                ),
                "Сравнительная степень, %": round(
                    100
                    * (comparison_degrees["Сравнительная"])
                    / (
                        comparison_degrees["Сравнительная"] + comparison_degrees["Превосходная"]
                        if (comparison_degrees["Сравнительная"] + comparison_degrees["Превосходная"]) != 0
                        else 1
                    ),
                    2,
                ),
                "Наречия, %": round(100 * counter["Наречие"] / total, 2),
                "Причастия, %": round(100 * counter["Причастие"] / total, 2),
                "Деепричастия, %": round(100 * counter["Деепричастие"] / total, 2),
                "Наклонения, %": round(
                    (naklonenie["Изъявительные"] + naklonenie["Повелительные"] + naklonenie["Условные"])
                    / (counter["Глаголы"] if counter["Глаголы"] != 0 else 1),
                    2,
                ),
                "Изъявительное наклонение, %": round(
                    100 * naklonenie["Изъявительные"] / (sum(naklonenie.values()) if sum(naklonenie.values()) != 0 else 1), 2
                ),
                "Условное наклонение, %": round(100 * naklonenie["Условные"] / (sum(naklonenie.values()) if sum(naklonenie.values()) != 0 else 1), 2),
                "Повелительное наклонение, %": round(
                    100 * naklonenie["Повелительные"] / (sum(naklonenie.values()) if sum(naklonenie.values()) != 0 else 1), 2
                ),
                "Среднее количество слов в предложении": round(alpha_counter["text_len"] / number_sentences, 2),
                "Средняя доля именованных сущностей в предложении": round(sum(alpha_counter["name_counter"]) / number_sentences, 2), 

                "Среднее количество морфем в слове": round(alpha_counter["morph_counter"] / total, 2),
                "Сентименты каждой трети текста и по тексту целиком": tuple(sentiments),
                "Средняя частота TOP10 самых редких слов в тексте": round(alpha_counter["avg10_freq_of_rare_words"], 2),
                "Среднее число синтаксических веток (по предложениям)": round(syntax_branches_counter["Количество синтаксических веток"] / number_sentences, 2),
                "Среднее число синтаксических веток (на одно слово)": round(syntax_branches_counter["Среднее число веток на одно слово"], 2),
                "Средняя глубина синтаксической ветки": round(syntax_branches_counter["Средняя глубина ветки"], 2),
                "Среднее число прилагательных в синтаксической ветке": round(syntax_branches_counter["Среднее число прилагательных в ветке"], 2),
                "Среднее число причастий в синтаксической ветке": round(syntax_branches_counter["Среднее число причастий в ветке"], 2),
                "Среднее число деепричастий в синтаксической ветке": round(syntax_branches_counter["Среднее число деепричастий в ветке"], 2),

                "Водность текста": round(1 - len(alpha_counter["list_important_words"]) / total, 2),
                "Вариабельность словаря": round(len(alpha_counter["list_unique_words"]) / total, 2),
                "Доля повторов в тексте": round(len(alpha_counter["list_repeated_words"]) / total, 2),
                "Доля местоимений Я": round(alpha_counter["input_word_frequency"]["я"] / total, 2),
                "Классическая тошнота": round(classic_sickness / total, 2),
                "Академическая тошнота": round(sum_top_freq / total, 2),
                "Доля местоимений в тексте": round(len(alpha_counter["list_pronouns"]) / total, 2),
                "Доля частиц в тексте": round(len(alpha_counter["list_parts"]) / total, 2),
                "Собственные/Нарицательные": round(counter["Собственные"] / counter["Нарицательные"] \
                                                   if counter["Нарицательные"] != 0 else 1, 2),
                "Доля числительных в тексте": round(len(alpha_counter["list_num"]) / total, 2),
                "Доля междометий в тексте": round(len(alpha_counter["list_intj"]) / total, 2),
                "Доля союзов в тексте": round(len(alpha_counter["list_conj"]) / total, 2),
                "Доля предлогов в тексте": round(len(alpha_counter["list_prep"]) / total, 2),


            }

        except:
            print(sys.exc_info())
            dict_res = defaultdict()

        return dict_res

    def df_of_text_features(self, text):
        dict_res = self.dict_of_text_features(text)
        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = pd.DataFrame.from_dict(dict_res)
        return df

    def get_table_of_morph_compare_with_sample(self, text, person_morph_df, sample_df):
        #name = ["Иван", "Иванов"]
        df = person_morph_df.loc[person_morph_df["Текст"] == text].head(1)
        m = self.m
        analysis = m.analyze(text)
        list_of_inf = [] #список глаголов в инфинитиве
        list_of_present = []
        list_of_past = []
        list_of_future = []
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if (
                    (speech_part == "V")
                    and not (TextProcessor.is_in_any("прич", params_list))
                    and not (TextProcessor.is_in_any("деепр", params_list))
                ):
                    if TextProcessor.is_in_any("прош", params_list) or "прош" in part:
                        text = text.replace(word, colored(word, "red"))
                        list_of_past.append(word)
                    elif TextProcessor.is_in_any("непрош", params_list) and ("сов" in part or TextProcessor.is_in_any("сов", params_list)):
                        text = text.replace(word, colored(word, "blue"))
                        list_of_future.append(word)
                    elif (
                        TextProcessor.is_in_any("непрош", params_list)
                        and not ("сов" in part or TextProcessor.is_in_any("сов", params_list))
                        and not ("несов" in part or TextProcessor.is_in_any("несов", params_list))
                    ):
                        text = text.replace(word, colored(word, "blue"))
                        list_of_future.append(word)
                    elif TextProcessor.is_in_any("наст", params_list) or TextProcessor.is_in_any("непрош", params_list):
                        text = text.replace(word, colored(word, "green"))
                        list_of_present.append(word)
                    elif TextProcessor.is_in_any("инф", params_list):
                        text = text.replace(word, colored(word, "white"))
                        list_of_inf.append(word)

        list_of_words = []
        list_of_words.append(list_of_past)
        list_of_words.append(list_of_present)
        list_of_words.append(list_of_future)
        list_of_words.append(list_of_inf)

        
        #text = text.replace("\n", " ")
        mean = np.round(sample_df.mean(), 1)
        std = np.round(sample_df.std(), 1)
        median = np.round(sample_df.median(), 1)
        q_20 = np.round(sample_df.quantile(q=0.2), 1)
        q_80 = np.round(sample_df.quantile(q=0.8), 1)
        df = df.append(pd.DataFrame(mean).T, ignore_index=True)
        df = df.append(pd.DataFrame(std).T, ignore_index=True)
        df = df.append(pd.DataFrame(median).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_20).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_80).T, ignore_index=True)
        df = df.T
        df = df.dropna()
        list_of_labels = [
            "Доля в тексте",
            "Cреднее по выборке",
            "STD по выборке",
            "Медиана по выборке",
            "20% квантиль по выборке",
            "80% квантиль по выборке",
        ]
        df = df.set_axis(list_of_labels, axis=1)
        df = df.rename_axis("Время").reset_index()
        df = df.round({'Среднее по выборке': 1, 'STD по выборке': 1, "Медиана по выборке": 1, "20% квантиль по выборке": 1,
            "80% квантиль по выборке": 1})
        df["Слова"] = pd.Series(list_of_words)

        for line in w.wrap(text): 
            print(line)

        return df


    def get_table_of_semantic_role_compare_with_sample(self, text, person_public_semantic_role_df, sample_df):
        
        
        df = person_public_semantic_role_df.loc[person_public_semantic_role_df["Текст"] == text].head(1)
        list_of_agences = df["Агенс, конструкции"].to_list()
        list_of_not_agences = df["Пациенс, конструкции"].to_list()
        list_of_internal_predicates = df["Внутренний предикат, конструкции"].to_list()
        list_of_external_predicates = df["Внешний предикат, конструкции"].to_list()
        list_of_words = []

        list_of_agences_ind = df["Агенс, индексы"].to_list()
        list_of_not_agences_ind = df["Пациенс, индексы"].to_list()
        list_of_internal_predicates_ind = df["Внутренний предикат, индексы"].to_list()
        list_of_external_predicates_ind = df["Внешний предикат, индексы"].to_list()
        if len(list_of_agences[0]) > 0:
            list_of_words.append(list_of_agences[0][0])
            list_of_agences_ind = list_of_agences_ind[0][0]
        else:
            list_of_words.append(list_of_agences[0])
        if len(list_of_not_agences[0]) > 0:
            list_of_words.append(list_of_not_agences[0][0])
            list_of_not_agences_ind = list_of_not_agences_ind[0][0]
        else:
            list_of_words.append(list_of_not_agences[0])
        if len(list_of_internal_predicates[0]) > 0:
            list_of_words.append(list_of_internal_predicates[0][0])
            list_of_internal_predicates_ind = list_of_internal_predicates_ind[0][0]
        else:
            list_of_words.append(list_of_internal_predicates[0])
        if len(list_of_external_predicates[0]) > 0:
            list_of_words.append(list_of_external_predicates[0][0])
            list_of_external_predicates_ind = list_of_external_predicates_ind[0][0]
        else:
            list_of_words.append(list_of_external_predicates[0])

        mean = np.round(sample_df.mean(), 1)
        std = np.round(sample_df.std(), 1)
        median = np.round(sample_df.median(), 1)
        q_20 = np.round(sample_df.quantile(q=0.2), 1)
        q_80 = np.round(sample_df.quantile(q=0.8), 1)
        df = df.append(pd.DataFrame(mean).T, ignore_index=True)
        df = df.append(pd.DataFrame(std).T, ignore_index=True)
        df = df.append(pd.DataFrame(median).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_20).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_80).T, ignore_index=True)
        df = df.T
        df = df.dropna()
        list_of_labels = [
            "Количество в тексте",
            "Cреднее по выборке",
            "STD по выборке",
            "Медиана по выборке",
            "20% квантиль по выборке",
            "80% квантиль по выборке",
        ]
        df = df.set_axis(list_of_labels, axis=1)
        df = df.rename_axis("Конструкция").reset_index()
        df = df.round({'Среднее по выборке': 1, 'STD по выборке': 1, "Медиана по выборке": 1, "20% квантиль по выборке": 1,
            "80% квантиль по выборке": 1})
        df["Словосочетания"] = pd.Series(list_of_words)

        color_text1 = self.color_text_by_2lists_of_ind(text, list_of_agences_ind, list_of_not_agences_ind, "red", "blue")

        for line in w.wrap(color_text1): 
            print(line)
        print("\n\n\n")
        color_text2 = self.color_text_by_2lists_of_ind(text, list_of_internal_predicates_ind, list_of_external_predicates_ind, "green", "magenta")
        for line in w.wrap(color_text2): 
            print(line)

        return df

    def get_table_of_report_compare_with_sample(self, text, person_public_df, sample_df):

        df = person_public_df.loc[person_public_df["Текст"] == text].head(1)
        mean = np.round(sample_df.mean(), 1)
        std = np.round(sample_df.std(), 1)
        median = np.round(sample_df.median(), 1)
        q_20 = np.round(sample_df.quantile(q=0.2), 1)
        q_80 = np.round(sample_df.quantile(q=0.8), 1)
        df = df.append(pd.DataFrame(mean).T, ignore_index=True)
        df = df.append(pd.DataFrame(std).T, ignore_index=True)
        df = df.append(pd.DataFrame(median).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_20).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_80).T, ignore_index=True)
        df = df.T
        df = df.dropna()
        list_of_labels = [
            "Значение фичи по тексту",
            "Cреднее по выборке",
            "STD по выборке",
            "Медиана по выборке",
            "20% квантиль по выборке",
            "80% квантиль по выборке",
        ]
        df = df.set_axis(list_of_labels, axis=1)
        df = df.rename_axis("Название фичи").reset_index()
        df = df.round({'Среднее по выборке': 1, 'STD по выборке': 1, "Медиана по выборке": 1, "20% квантиль по выборке": 1,
            "80% квантиль по выборке": 1})

        # красим текст, в зависимости от частей речи, выводимый перед таблицей с фичами
        text = self.color_part_of_speech_in_text(text)

        for line in w.wrap(text): 
            print(line)
        return df

    def color_part_of_speech_in_text(self, text):
        m = self.m
        analysis = m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                word = token_data["text"]
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if speech_part == "V":

                    if "прич" in part or TextProcessor.is_in_any("прич", params_list):
                        text = re.sub(r"\b" + word, colored(word, "blue"), text)
                        continue

                    elif "деепр" in part or TextProcessor.is_in_any("деепр", params_list):
                        text = re.sub(r"\b" + word, colored(word, "white"), text)
                        continue

                    if "сов" in part or TextProcessor.is_in_any("сов", params_list):
                        text = re.sub(r"\b" + word, colored(word, "magenta"), text)  # совершенный вид
                    elif "несов" in part or TextProcessor.is_in_any("несов", params_list):
                        text = re.sub(r"\b" + word, colored(word, "cyan"), text)  # несовершенный вид
                    else:
                        logging.info("Проблема с глаголами", token_data["analysis"])

                elif speech_part == "A":
                    text = re.sub(r"\b" + word, colored(word, "green"), text)  # прилагательное

                elif speech_part == "ADV":
                    text = re.sub(r"\b" + word, colored(word, "yellow"), text)  # наречие

                elif speech_part == "S":
                    text = re.sub(r"\b" + word, colored(word, "red"), text)  # существительное

        return text

    def color_verbs_times_in_text(self, text):
        m = self.m
        analysis = m.analyze(text)

        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if (
                    (speech_part == "V")
                    and not (TextProcessor.is_in_any("прич", params_list))
                    and not (TextProcessor.is_in_any("деепр", params_list))
                ):
                    if TextProcessor.is_in_any("прош", params_list) or "прош" in part:
                        text = text.replace(word, colored(word, "red"))

                    elif TextProcessor.is_in_any("непрош", params_list) and ("сов" in part or TextProcessor.is_in_any("сов", params_list)):
                        text = text.replace(word, colored(word, "blue"))
       
                    elif (
                        TextProcessor.is_in_any("непрош", params_list)
                        and not ("сов" in part or TextProcessor.is_in_any("сов", params_list))
                        and not ("несов" in part or TextProcessor.is_in_any("несов", params_list))
                    ):
                        text = text.replace(word, colored(word, "blue"))
  
                    elif TextProcessor.is_in_any("наст", params_list) or TextProcessor.is_in_any("непрош", params_list):
                        text = text.replace(word, colored(word, "green"))

                    elif TextProcessor.is_in_any("инф", params_list):
                        text = text.replace(word, colored(word, "white"))

        return text

    def color_semantic_role_in_text(self, text, person_public_semantic_role_df):

        df = person_public_semantic_role_df.loc[person_public_semantic_role_df["Текст"] == text].head(1)
        list_of_agences = df["Агенс, конструкции"].to_list()
        list_of_not_agences = df["Пациенс, конструкции"].to_list()
        list_of_internal_predicates = df["Внутренний предикат, конструкции"].to_list()
        list_of_external_predicates = df["Внешний предикат, конструкции"].to_list()


        list_of_agences_ind = df["Агенс, индексы"].to_list()
        list_of_not_agences_ind = df["Пациенс, индексы"].to_list()
        list_of_internal_predicates_ind = df["Внутренний предикат, индексы"].to_list()
        list_of_external_predicates_ind = df["Внешний предикат, индексы"].to_list()
        if len(list_of_agences[0]) > 0:
            list_of_agences_ind = list_of_agences_ind[0][0]

        if len(list_of_not_agences[0]) > 0:
            list_of_not_agences_ind = list_of_not_agences_ind[0][0]

        if len(list_of_internal_predicates[0]) > 0:
            list_of_internal_predicates_ind = list_of_internal_predicates_ind[0][0]

        if len(list_of_external_predicates[0]) > 0:
            list_of_external_predicates_ind = list_of_external_predicates_ind[0][0]

        color_text1 = self.color_text_by_2lists_of_ind(text, list_of_agences_ind, list_of_not_agences_ind, "red", "blue")
        color_text2 = self.color_text_by_2lists_of_ind(text, list_of_internal_predicates_ind, list_of_external_predicates_ind, "green", "magenta")

        return color_text1 + "\n\n\n" + color_text2


    @staticmethod
    def plot_one(text, person_public_df, public_df):
        list_of_text_features = public_df._get_numeric_data().columns.values.tolist()
        list_of_text_features.reverse()
        n = len(list_of_text_features)
        #plt.rcParams["figure.figsize"] = (18,n)
        plt.rcParams["figure.figsize"] = (24, n)
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        fig, ax = plt.subplots((n // 2 + n % 2), 2)
        for i in range(n // 2 + n % 2):
            for j in range(2):
                try:
                    feature_name = list_of_text_features.pop()
                except:
                    break
                feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
                num, rang, psyh = size_range_psyh_of_feature_value(text, feature_name, person_public_df, public_df)
                x = public_df[f"{feature_name}"].to_list()
                sns.kdeplot(data=x, fill = True, ax = ax[i][j], palette = pal, cut = 0)
                ax[i][j].set_title(f"{feature_name}", y=1.0, pad=-14, fontsize=16)
                ax[i][j].set_xlabel(f'Значение признака', fontsize=13)
                ax[i][j].set_ylabel(f'Доля людей', fontsize=13)
                ax[i][j].axvline(x=np.quantile(x, q=0.2), c='deepskyblue')
                ax[i][j].axvline(x=np.quantile(x, q=0.5), c='dodgerblue')
                ax[i][j].axvline(x=np.quantile(x, q=0.8), c='royalblue')
                ax[i][j].axvline(x=np.mean(x), c='y')
                ax[i][j].axvline(x=feature_value, label=f'{num} = {feature_value}\n{psyh}', c='red')
                
                ax[i][j].legend(fontsize = 12, loc='right')
        fig.tight_layout()
        plt.show()


    def visualize_text(self, text, person_public_df, public_df):
        # list_of_text_features = ['Доля коротких предложений', 'Доля длинных предложений', 'Доля длинных низкочастотных слов', 'Доля коротких высокочастотных слов', 'Доля восклицательных предложений', 'Доля вопросительных предложений', 'Доля частиц НЕ', 'Гласные, %', 'Согласные, %', 'Существительные, %', 'Глаголы, %', 'Прилагательные, %', 'Наречия, %', 'Причастия, %', 'Деепричастия, %', 'Повелительное наклонение, %', 'Доля предложений на одно слово', 'Доля сложноподчиненных предложений', 'Средняя длина предложения', 'Cредняя длинна слова в тексте', 'Среднее число запятых в предложении', 'Гласная "И", %', 'Гласная "О", %', 'Гласная "А", %', 'Совершенный вид, %', 'Несовершенный вид, %', 'Превосходная степень, %', 'Сравнительная степень, %', 'Изъявительное наклонение, %', 'Условное наклонение, %', 'Среднее количество слов в предложении', 'Средняя доля именованных сущностей в предложении', 'Среднее количество именованных сущностей в предложении', 'Среднее количество морфем в слове', 'Средняя частота TOP10 самых редких слов в тексте', 'Среднее число синтаксических веток (по предложениям)', 'Среднее число синтаксических веток (на одно слово)', 'Средняя глубина синтаксической ветки', 'Среднее число прилагательных в синтаксической ветке', 'Среднее число причастий в синтаксической ветке', 'Среднее число деепричастий в синтаксической ветке']
        print(self.color_part_of_speech_in_text(text))
        self.plot_one(text, person_public_df, public_df)
        print("\n\n")


    def visualize_text_morph(self, text, person_public_df, public_df):
	    print(self.color_verbs_times_in_text(text))
	    self.plot_one(text, person_public_df, public_df)
	    print("\n\n")


    def visualize_text_semantic_role(self, text, person_public_df, public_df):
        print(self.color_semantic_role_in_text(text, person_public_df))
        self.plot_one(text, person_public_df, public_df)
        print("\n\n")


    def table_by_text_psyh_cosy(self, text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
        for line in w.wrap(self.color_part_of_speech_in_text(text)): 
            print(line)
        table = df_with_features_ranges_and_psych_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df)
        display(table)

        
    def dict_with_conceptual(self, text):
        person_public_dict_with_distances = {}

        dict_res = {}
        df_dist_by_words = distance_between_text_and_psychotypes(text, self.elmo_model, self.nlp_core)
        df_ist, df_shiz, df_epi, df_emo = processed_df_with_dist(df_dist_by_words)

        dict_res["Истероид"] = {}
        dict_res["Шизоид"] = {}
        dict_res["Эпилептоид"] = {}
        dict_res["Эмотив"] = {}

        if not np.isnan(np.mean(df_ist["Истероид"])):
            dict_res["Истероид"]["group_mean"] = np.mean(df_ist["Истероид"])
            dict_res["Истероид"]["count"] = len(df_ist["Истероид"])
            dict_res["Истероид"]["Слова"] = df_ist['Слово'].to_list()
        else:
            dict_res["Истероид"]["group_mean"] = 1
            dict_res["Истероид"]["count"] = len(df_ist["Истероид"])
            dict_res["Истероид"]["Слова"] = []

        if not np.isnan(np.mean(df_shiz["Шизоид"])):
            dict_res["Шизоид"]["group_mean"] = np.mean(df_shiz["Шизоид"])
            dict_res["Шизоид"]["count"] = len(df_shiz["Шизоид"])
            dict_res["Шизоид"]["Слова"] = df_shiz['Слово'].to_list()
        else:
            dict_res["Шизоид"]["group_mean"] = 1
            dict_res["Шизоид"]["count"] = len(df_shiz["Шизоид"])
            dict_res["Шизоид"]["Слова"] = []

        if not np.isnan(np.mean(df_epi["Эпилептоид"])):
            dict_res["Эпилептоид"]["group_mean"] = np.mean(df_epi["Эпилептоид"])
            dict_res["Эпилептоид"]["count"] = len(df_epi["Эпилептоид"])
            dict_res["Эпилептоид"]["Слова"] = df_epi['Слово'].to_list()
        else:
            dict_res["Эпилептоид"]["group_mean"] = 1
            dict_res["Эпилептоид"]["count"] = len(df_epi["Эпилептоид"])
            dict_res["Эпилептоид"]["Слова"] = []

        if not np.isnan(np.mean(df_emo["Эмотив"])):
            dict_res["Эмотив"]["group_mean"] = np.mean(df_emo["Эмотив"])
            dict_res["Эмотив"]["count"] = len(df_emo["Эмотив"])
            dict_res["Эмотив"]["Слова"] = df_emo['Слово'].to_list()
        else:
            dict_res["Эмотив"]["group_mean"] = 1
            dict_res["Эмотив"]["count"] = len(df_emo["Эмотив"])
            dict_res["Эмотив"]["Слова"] = []

        person_public_dict_with_distances[text] = dict_res
        return person_public_dict_with_distances
        
    def psychotype_by_text_elmo(self, text, dict_with_conceptual):
        dict_temp = dict_with_conceptual[text]
        
        dict_count = {}
        for psych in dict_temp.keys():
            dict_count[psych] = dict_temp[psych]["count"]
        
        dict_group_mean = {}
        for psych in dict_temp.keys():
            dict_group_mean[psych] = dict_temp[psych]["group_mean"]
            
        dict_count_res = {k[0]: [k[1]] for k in dict_count.items()}
        
        dict_group_mean_res = {k[0]: [k[1]] for k in dict_group_mean.items()}
        display(Markdown("## * __Значение = Количество слов, близких к психотипу__"))
        display(pd.DataFrame.from_dict(dict_count_res))
        print("\n")
        display(Markdown("## * __Значение = Среднее расстояние от слов, близких к данному психотипу__"))
        display(pd.DataFrame.from_dict(dict_group_mean_res))
        print("\n")

    def compute_one_df(self, text, df_name):
        df = pd.DataFrame()
        if df_name == "public_df":
            df = df.append(self.df_of_text_features(text), ignore_index=True)
            return df
        elif df_name == "public_semantic_role_df":
            df = df.append(self.df_of_semantic_role_features(text), ignore_index=True)
            return df
        elif df_name == "morph_df":
            df = df.append(self.df_of_morphology_features(text), ignore_index=True)
            return df
        elif df_name == "conceptual_dict":
            df = self.dict_with_conceptual(text)
            return df
        elif df_name == "public_modality_df":
            df = df.append(self.df_of_modality_features(text), ignore_index=True)
        else:
            print("Неверное имя datafram'а")
        return df
        
    def df_of_modality_features(self, text):
        dict_res = self.dict_of_modality_features(text)
        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = np.round(pd.DataFrame.from_dict(dict_res), 1)
        return df
    
    def dict_of_modality_features(self, text):
    
        dict_words_inds = self.dict_of_modality(text)
        list_of_opt = dict_words_inds["Оптативная"]
        list_of_hyp_pot = dict_words_inds["Гипотетическая и потенциальная"]
        list_of_int = dict_words_inds["Интенция"]
        list_of_inp_deb = dict_words_inds["Императив и дебитив"]

        dict_res = {"Текст": text,
        "Оптативная" : len(list_of_opt[0]), 
        "Гипотетическая и потенциальная": len(list_of_hyp_pot[0]),
        "Интенция" : len(list_of_int[0]), 
        "Императив и дебитив" : len(list_of_inp_deb[0]), 
          "Оптативная, конструкции": list_of_opt[0],
          "Гипотетическая и потенциальная, конструкции": list_of_hyp_pot[0],
            "Интенция, конструкции": list_of_int[0],
            "Императив и дебитив, конструкции": list_of_inp_deb[0],
                  "Оптативная, индексы": list_of_opt[1],
          "Гипотетическая и потенциальная, индексы": list_of_hyp_pot[1],
            "Интенция, индексы": list_of_int[1],
            "Императив и дебитив, индексы": list_of_inp_deb[1],
          }

        return dict_res
        
    def list_of_future_and_inf_verbs(self, text):
        list_of_inf_words = []
        list_of_fut_words = []
        list_of_words = []
        list_of_perf_f = []
        list_of_imperf_f = []
        analysis = self.m.analyze(text)
        for token_data in analysis:
            if "analysis" in token_data:
                if len(token_data["analysis"]) != 1:
                    continue
                word = token_data["text"]
                part, params = token_data["analysis"][0]["gr"].split("=")
                if isinstance(params, str) and params.startswith("("):
                    params = params[1:-1]

                part = part.split(",")
                speech_part = part[0]

                params_list = params.split("|")
                params_list = [x.split(",") for x in params_list]

                if (
                    (speech_part == "V")
                    and not (TextProcessor.is_in_any("прич", params_list))
                    and not (TextProcessor.is_in_any("деепр", params_list))
                ):
                    list_of_words.append(word)
                    if "сов" in part or TextProcessor.is_in_any("сов", params_list):
                        list_of_perf_f.append(word)
                    elif "несов" in part or TextProcessor.is_in_any("несов", params_list):
                        list_of_imperf_f.append(word)

                    if TextProcessor.is_in_any("непрош", params_list) and ("сов" in part or TextProcessor.is_in_any("сов", params_list)):
                        list_of_fut_words.append(word)
                    elif (
                        TextProcessor.is_in_any("непрош", params_list)
                        and not ("сов" in part or TextProcessor.is_in_any("сов", params_list))
                        and not ("несов" in part or TextProcessor.is_in_any("несов", params_list))
                    ):
                        list_of_fut_words.append(word)
                    elif TextProcessor.is_in_any("инф", params_list):
                        list_of_inf_words.append(word)
        return list_of_fut_words, list_of_inf_words, list_of_words, list_of_perf_f, list_of_imperf_f
        
    def dict_of_modality(self, text):
    
        list_of_verbs = self.list_of_future_and_inf_verbs(text)

        dict_res = {}

        list_of_opt = []
        list_of_opt_inds = []

        list_of_hyp_pot = []
        list_of_hyp_pot_inds = []

        list_of_int = []
        list_of_int_inds = []

        list_of_inp_deb = []
        list_of_inp_deb_inds = []

        doc = self.nlp_core(text)
        i = 0
        for sent in doc.sentences:
            
            flag_opt = False
            flag_hyp_pot = False
            flag_int = False
            flag_inp_deb = False
            
            for token in sent.tokens:
                if not token.words:
                    continue
                token = token.words[0]
                # оптативная
                if token.text.lower() in ["бы", "хочу", "хотел", "хотела"]:
                    list_of_opt.append(token.text)
                    list_of_opt_inds.append(i)
                    flag_opt = True
                # гипотетическая и потенциальная
                elif token.text.lower() in ["возможно", "можно", "мог", "могла", "могли", "наверное", "может", "могут", "где-нибудь", "кажется", "ли", "ль"] or "-то" in token.text.lower() or "-либо" in token.text.lower() or "-нибудь" in token.text.lower() or (token.text in list_of_verbs[2] and token.text.lower().startswith("по") and not ("бы" in sent.text.lower() or "хочу" in sent.text.lower() or "хотел" in sent.text.lower() or "хотела" in sent.text.lower())) or (token.text.lower() == "скорее" and "всего" in sent.text.lower()):
                    list_of_hyp_pot.append(token.text)
                    list_of_hyp_pot_inds.append(i)
                    flag_hyp_pot = True
                # интенция
                elif not (flag_hyp_pot) and (token.text in list_of_verbs[0] and token.text in list_of_verbs[4] or token.text.lower() == "чтобы" or token.text.lower() == "интересно" or token.text.lower() == "если"):
                    list_of_int.append(token.text)
                    list_of_int_inds.append(i)
                    flag_int = True
                # Императив и дебитив
                elif ((token.text.lower() in ["должен", "должна", "надо", "необходимо", "необходима", "необходим", "необходимы",  "нужно", "нужен", "нужна", "требуется", "требуются", "же"]) and ("ли" not in sent.text.lower()) and ("если" not in sent.text.lower()) or (token.text in list_of_verbs[3] and not token.text.lower().startswith("по") and not flag_hyp_pot and ("если" not in sent.text.lower())) ) and not ("бы" in sent.text.lower() or "хочу" in sent.text.lower() or "хотел" in sent.text.lower() or "хотела" in sent.text.lower()):
                    list_of_inp_deb.append(token.text)
                    list_of_inp_deb_inds.append(i)
                    flag_inp_deb = True
                i += 1

        dict_res["Оптативная"] = (list_of_opt, list_of_opt_inds)
        dict_res["Гипотетическая и потенциальная"] = (list_of_hyp_pot, list_of_hyp_pot_inds)
        dict_res["Интенция"] = (list_of_int, list_of_int_inds)
        dict_res["Императив и дебитив"] = (list_of_inp_deb, list_of_inp_deb_inds)
        return dict_res
        
    def df_of_relative_features(self, glue_text, person_public_df, person_public_semantic_role_df, person_public_modality_df):
        dict_res = {}
        dict_res["Текст"] = glue_text
        dict_res["Доля частиц НЕ на ответ"] = person_public_df[person_public_df["Текст"] == glue_text]["Доля частиц НЕ"].values[0]
        dict_res["Доля агенсов на ответ"] = person_public_semantic_role_df[person_public_semantic_role_df["Текст"] == glue_text]["Агенс"].values[0]
        dict_res["Наречия, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Наречия, %"].values[0]
        dict_res["Деепричастия, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0]
        dict_res["Причастия, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Причастия, %"].values[0]
        dict_res["Прилагательные, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Прилагательные, %"].values[0]
        dict_res["Глаголы, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Глаголы, %"].values[0]
        dict_res["Совершенный вид, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Совершенный вид, %"].values[0]
        dict_res["Несовершенный вид, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Несовершенный вид, %"].values[0]
        dict_res["Количество коротких предложений"] = person_public_df[person_public_df["Текст"] == glue_text]["Доля коротких предложений"].values[0]
        dict_res["Количество длинных предложений"] = person_public_df[person_public_df["Текст"] == glue_text]["Доля длинных предложений"].values[0]
        dict_res["Гласные, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Гласные, %"].values[0]
        dict_res["Согласные, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Согласные, %"].values[0]
        dict_res["Условное наклонение, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Условное наклонение, %"].values[0]
        dict_res["Повелительное наклонение, %"] = person_public_df[person_public_df["Текст"] == glue_text]["Повелительное наклонение, %"].values[0]



        if person_public_modality_df[person_public_modality_df["Текст"] == glue_text]["Оптативная"].values[0] > 0:
            dict_res["Оптативная"] = 1
        else:
            dict_res["Оптативная"] = 0

        if person_public_modality_df[person_public_modality_df["Текст"] == glue_text]["Гипотетическая и потенциальная"].values[0] > 0:
            dict_res["Гипотетическая и потенциальная"] = 1
        else:
            dict_res["Гипотетическая и потенциальная"] = 0

        if person_public_modality_df[person_public_modality_df["Текст"] == glue_text]["Интенция"].values[0] > 0:
            dict_res["Интенция"] = 1
        else:
            dict_res["Интенция"] = 0

        if person_public_modality_df[person_public_modality_df["Текст"] == glue_text]["Императив и дебитив"].values[0] > 0:
            dict_res["Императив и дебитив"] = 1
        else:
            dict_res["Императив и дебитив"] = 0


        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = pd.DataFrame.from_dict(dict_res)
        return df


    def table_by_text_modality(self, text, person_public_modality_df, public_modality_df):
        table = self.get_table_of_modality_compare_with_sample(text, person_public_modality_df, public_modality_df)
        display(table)
        print("\n")


    def get_table_of_modality_compare_with_sample(self, text, person_public_modality_df, public_modality_df):
        df = person_public_modality_df.loc[person_public_modality_df["Текст"] == text].head(1)
        
        list_of_opt = df["Оптативная, конструкции"].to_list()
        list_of_hyp_pot = df["Гипотетическая и потенциальная, конструкции"].to_list()
        list_of_int = df["Интенция, конструкции"].to_list()
        list_of_inp_deb = df["Императив и дебитив, конструкции"].to_list()
        
        list_of_opt_inds = df["Оптативная, индексы"].to_list()
        list_of_hyp_pot_inds = df["Гипотетическая и потенциальная, индексы"].to_list()
        list_of_int_inds = df["Интенция, индексы"].to_list()
        list_of_inp_deb_inds = df["Императив и дебитив, индексы"].to_list()
        
        
        list_of_words = []

        list_of_words.append(list_of_opt)
        list_of_words.append(list_of_hyp_pot)
        list_of_words.append(list_of_int)
        list_of_words.append(list_of_inp_deb)

        mean = np.round(public_modality_df.mean(), 1)
        std = np.round(public_modality_df.std(), 1)
        median = np.round(public_modality_df.median(), 1)
        q_20 = np.round(public_modality_df.quantile(q=0.2), 1)
        q_80 = np.round(public_modality_df.quantile(q=0.8), 1)
        df = df.append(pd.DataFrame(mean).T, ignore_index=True)
        df = df.append(pd.DataFrame(std).T, ignore_index=True)
        df = df.append(pd.DataFrame(median).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_20).T, ignore_index=True)
        df = df.append(pd.DataFrame(q_80).T, ignore_index=True)
        df = df.T
        df = df.dropna()
        list_of_labels = [
            "Количество в тексте",
            "Cреднее по выборке",
            "STD по выборке",
            "Медиана по выборке",
            "20% квантиль по выборке",
            "80% квантиль по выборке",
        ]
        df = df.set_axis(list_of_labels, axis=1)
        df = df.rename_axis("Конструкция").reset_index()
        df = df.round({'Среднее по выборке': 1, 'STD по выборке': 1, "Медиана по выборке": 1, "20% квантиль по выборке": 1,
            "80% квантиль по выборке": 1})
        df["Слова"] = pd.Series(list_of_words)

        color_text1 = self.color_list_of_stanza_inds_in_text_for_modality(text, list_of_opt_inds[0], list_of_hyp_pot_inds[0], list_of_int_inds[0], list_of_inp_deb_inds[0], "red", "green", "yellow", "blue")
        for line in w.wrap(color_text1): 
            print(line)

        return df

    def color_list_of_stanza_inds_in_text_for_modality(self, text, list_of_opt_inds, list_of_hyp_pot_inds, list_of_int_inds, list_of_inp_deb_inds, color1, color2, color3, color4):
        doc = self.nlp_core(text)
        color_text = ""
        i = 0
        for sent in doc.sentences:
            for token in sent.tokens:
                if not token.words:
                    continue
                token = token.words[0]
                if i in list_of_opt_inds:
                    color_text += colored(token.text, color1)
                    color_text += " "
                elif i in list_of_hyp_pot_inds:
                    color_text += colored(token.text, color2)
                    color_text += " "
                elif i in list_of_int_inds:
                    color_text += colored(token.text, color3)
                    color_text += " "
                elif i in list_of_inp_deb_inds:
                    color_text += colored(token.text, color4)
                    color_text += " "
                else:
                    color_text += token.text
                    color_text += " "
                i += 1
        res_text = TextProcessor.return_the_style_of_text(color_text)
        return res_text
        
    def df_with_new_labels_by_text(self, glue_text, person_public_df, public_df):

        dict_res = {}
        dict_res["Текст"] = glue_text
        
        processed_text = self.nlp_core(glue_text)
        

        if person_public_df[person_public_df["Текст"] == glue_text]["Прилагательные, %"].values[0] > person_public_df[person_public_df["Текст"] == glue_text]["Глаголы, %"].values[0]:
            dict_res["Глаголы"] = 0
            dict_res["Прилагательные"] = 1
        elif person_public_df[person_public_df["Текст"] == glue_text]["Прилагательные, %"].values[0] < person_public_df[person_public_df["Текст"] == glue_text]["Глаголы, %"].values[0]:
            dict_res["Глаголы"] = 1
            dict_res["Прилагательные"] = 0
        else:
            dict_res["Глаголы"] = 1
            dict_res["Прилагательные"] = 1
        
        if person_public_df[person_public_df["Текст"] == glue_text]["Наречия, %"].values[0] > np.median(public_df["Наречия, %"]):
            dict_res["Наречия"] = 1
        else:
            dict_res["Наречия"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0] > np.median(public_df["Деепричастия, %"]):
            dict_res["Деепричастия"] = 1
        else:
            dict_res["Деепричастия"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Причастия, %"].values[0] > np.median(public_df["Причастия, %"]):
            dict_res["Причастия"] = 1
        else:
            dict_res["Причастия"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Совершенный вид, %"].values[0] > person_public_df[person_public_df["Текст"] == glue_text]["Несовершенный вид, %"].values[0]:
            dict_res["Совершенная форма"] = 1
            dict_res["Несовершенная форма"] = 0
        elif person_public_df[person_public_df["Текст"] == glue_text]["Совершенный вид, %"].values[0] < person_public_df[person_public_df["Текст"] == glue_text]["Несовершенный вид, %"].values[0]:
            dict_res["Совершенная форма"] = 0
            dict_res["Несовершенная форма"] = 1
        else:
            dict_res["Совершенная форма"] = 1
            dict_res["Несовершенная форма"] = 1
            
        if person_public_df[person_public_df["Текст"] == glue_text]["Количество коротких предложений"].values[0] > person_public_df[person_public_df["Текст"] == glue_text]["Количество длинных предложений"].values[0]:
            dict_res["Простые предложения"] = 1
            dict_res["Сложные предложения"] = 0
        elif person_public_df[person_public_df["Текст"] == glue_text]["Количество коротких предложений"].values[0] < person_public_df[person_public_df["Текст"] == glue_text]["Количество длинных предложений"].values[0]:
            dict_res["Простые предложения"] = 0
            dict_res["Сложные предложения"] = 1
        else:
            dict_res["Простые предложения"] = 1
            dict_res["Сложные предложения"] = 1
            
        if person_public_df[person_public_df["Текст"] == glue_text]["Согласные, %"].values[0] > np.median(public_df["Согласные, %"]):
            dict_res["Гласные"] = 0
            dict_res["Согласные"] = 1
        else:
            dict_res["Гласные"] = 1
            dict_res["Согласные"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Условное наклонение, %"].values[0] > 0:
            dict_res["Условное наклонение"] = 1
        else:
            dict_res["Условное наклонение"] = 0
            
        if person_public_df[person_public_df["Текст"] == glue_text]["Повелительное наклонение, %"].values[0] > 0:
            dict_res["Повелительное наклонение"] = 1
        else:
            dict_res["Повелительное наклонение"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Доля частиц НЕ на ответ"].values[0] > 0:
            dict_res["Частица НЕ"] = 1
        else:
            dict_res["Частица НЕ"] = 0


        if person_public_df[person_public_df["Текст"] == glue_text]["Оптативная"].values[0] > 0:
            dict_res["Оптативная"] = 1
        else:
            dict_res["Оптативная"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Гипотетическая и потенциальная"].values[0] > 0:
            dict_res["Гипотетическая и потенциальная"] = 1
        else:
            dict_res["Гипотетическая и потенциальная"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Интенция"].values[0] > 0:
            dict_res["Интенция"] = 1
        else:
            dict_res["Интенция"] = 0

        if person_public_df[person_public_df["Текст"] == glue_text]["Императив и дебитив"].values[0] > 0:
            dict_res["Императив и дебитив"] = 1
        else:
            dict_res["Императив и дебитив"] = 0


        if person_public_df[person_public_df["Текст"] == glue_text]["Доля агенсов на ответ"].values[0] > 0.1:
            dict_res["Агенс"] = 1
            dict_res["Пациенс"] = 0
        else:
            dict_res["Агенс"] = 0
            dict_res["Пациенс"] = 1

        list_rand = ["Прилагательные", "Наречия", "Причастия", "Несовершенная форма", "Гласные", "Условное наклонение", "Оптативная", "Гипотетическая и потенциальная"]
        list_syst = ["Глаголы", "Деепричастия", "Совершенная форма", "Согласные", "Повелительное наклонение", "Частица НЕ", "Интенция", "Императив и дебитив"]

        list_loc = ["Пациенс", "Глаголы", "Прилагательные", "Оптативная", "Императив и дебитив", "Простые предложения", "Частица НЕ"]
        list_glob = ["Агенс", "Наречия", "Деепричастия", "Причастия", "Гипотетическая и потенциальная", "Сложные предложения", "Интенция"]

        dict_res["Хаос"] = round(np.mean([dict_res[key] for key in list_rand]) * 100)
        dict_res["Система"] = round(np.mean([dict_res[key] for key in list_syst]) * 100)

        dict_res["Локальная"] = round(np.mean([dict_res[key] for key in list_loc]) * 100)
        dict_res["Глобальная"] = round(np.mean([dict_res[key] for key in list_glob]) * 100)

        if dict_res["Хаос"] > dict_res["Система"]:
            dict_res["Хаос / Система"] = 0
        elif dict_res["Хаос"] < dict_res["Система"]:
            dict_res["Хаос / Система"] = 2
        elif abs(dict_res["Хаос"] - dict_res["Система"]) < 1e-3:
            dict_res["Хаос / Система"] = 1

        if dict_res["Локальная"] > dict_res["Глобальная"]:
            dict_res["Локальная / Глобальная"] = 0
        elif dict_res["Локальная"] < dict_res["Глобальная"]:
            dict_res["Локальная / Глобальная"] = 2
        elif abs(dict_res["Локальная"] - dict_res["Глобальная"]) < 1e-3:
            dict_res["Локальная / Глобальная"] = 1
        
        dict_res = {k[0]: [k[1]] for k in dict_res.items()}
        df = pd.DataFrame.from_dict(dict_res)

        return df


    def dict_verbs_norm(self, text):
        dict_verbs_norm = defaultdict()
        for sentence in self.nlp_core(text).sentences:
            for token in sentence.tokens:
                if not token.words:
                    continue
                token = token.words[0]
                if "VerbForm=Inf" in str(token.feats) or "VerbForm=Fin" in str(token.feats):
                    dict_verbs_norm[token.text] = token.lemma
        return dict_verbs_norm

    def predicate_report(self, text, public_semantic_role_df):
        dict_with_norm = self.dict_verbs_norm(text)
        list_feel_verbs, list_mind_verbs, list_relation_verbs, list_modal_verbs, list_skills_verbs = verbs_df_to_internal_pred_verbs(self.verbs_df)
        list_action_verbs, list_communication_verbs, list_being_verbs, list_move_verbs = verbs_df_to_external_pred_verbs(self.verbs_df)
        verbs_from_text = list(dict_with_norm.keys())
        counter = Counter()
        # Внутренний предикат
        counter["Чувства"] = 0
        counter["Мысли"] = 0
        counter["Отношение"] = 0
        counter["Модальные"] = 0
        counter["Умения"] = 0
        list_feel_from_text = []
        list_mind_from_text = []
        list_relation_from_text = []
        list_modal_from_text = []
        list_skills_from_text = []
        # Внешний предикат
        counter["Действие"] = 0
        counter["Коммуникация"] = 0
        counter["Бытие"] = 0
        counter["Движение"] = 0
        list_action_from_text = []
        list_communication_from_text = []
        list_being_from_text = []
        list_move_from_text = []
        
        counter["Внутренний предикат"] = 0
        counter["Внешний предикат"] = 0
        list_of_internal_predicat = []
        list_of_external_predicat = []
        
        counter["Другие глаголы"] = 0
        list_of_otherwise_verbs = []

        for verb in verbs_from_text:
            if dict_with_norm.get(verb) in list_feel_verbs:
                counter["Чувства"] += 1
                list_feel_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_mind_verbs:
                counter["Мысли"] += 1
                list_mind_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_relation_verbs:
                counter["Отношение"] += 1
                list_relation_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_modal_verbs:
                counter["Модальные"] += 1
                list_modal_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_skills_verbs:
                counter["Умения"] += 1
                list_skills_from_text.append(verb)
            
            elif dict_with_norm.get(verb) in list_action_verbs:
                counter["Действие"] += 1
                list_action_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_communication_verbs:
                counter["Коммуникация"] += 1
                list_communication_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_being_verbs:
                counter["Бытие"] += 1
                list_being_from_text.append(verb)
                
            elif dict_with_norm.get(verb) in list_move_verbs:
                counter["Движение"] += 1
                list_move_from_text.append(verb)
            else:
                counter["Другие глаголы"] += 1
                list_of_otherwise_verbs.append(verb)

        
        list_of_internal_predicat = list_feel_from_text + list_mind_from_text + list_relation_from_text + list_modal_from_text + list_skills_from_text
        list_of_external_predicat = list_action_from_text + list_communication_from_text + list_being_from_text + list_move_from_text
        counter["Внутренний предикат"] = len(list_of_internal_predicat)
        counter["Внешний предикат"] = len(list_of_external_predicat)
        
        dict_feel = {"Категория": "Чувства", "Слова": list_feel_from_text, "Количество в тексте": counter["Чувства"]}
        dict_mind = {"Категория": "Мысли", "Слова": list_mind_from_text, "Количество в тексте": counter["Мысли"]}
        dict_relation = {"Категория": "Отношение", "Слова": list_relation_from_text, "Количество в тексте": counter["Отношение"]}
        dict_modal = {"Категория": "Модальные", "Слова": list_modal_from_text, "Количество в тексте": counter["Модальные"]}
        dict_skills = {"Категория": "Умения", "Слова": list_skills_from_text, "Количество в тексте": counter["Умения"]}
        dict_action = {"Категория": "Действие", "Слова": list_action_from_text, "Количество в тексте": counter["Действие"]}
        dict_communication = {"Категория": "Коммуникация", "Слова": list_communication_from_text, "Количество в тексте": counter["Коммуникация"]}
        dict_being = {"Категория": "Бытие", "Слова": list_being_from_text, "Количество в тексте": counter["Бытие"]}
        dict_move = {"Категория": "Движение", "Слова": list_move_from_text, "Количество в тексте": counter["Движение"]}
        dict_int_pred = {"Категория": "Внутренний предикат", "Слова": list_of_internal_predicat, "Количество в тексте": counter["Внутренний предикат"]}
        dict_ext_pred = {"Категория": "Внешний предикат", "Слова": list_of_external_predicat, "Количество в тексте": counter["Внешний предикат"]}
        dict_otherwise = {"Категория": "Другие глаголы", "Слова": list_of_otherwise_verbs, "Количество в тексте": counter["Другие глаголы"]}
        
        res_df = pd.DataFrame()
        res_df = res_df.append(dict_feel, ignore_index=True)
        res_df = res_df.append(dict_mind, ignore_index=True)
        res_df = res_df.append(dict_relation, ignore_index=True)
        res_df = res_df.append(dict_modal, ignore_index=True)
        res_df = res_df.append(dict_skills, ignore_index=True)
        res_df = res_df.append(dict_action, ignore_index=True)
        res_df = res_df.append(dict_communication, ignore_index=True)
        res_df = res_df.append(dict_being, ignore_index=True)
        res_df = res_df.append(dict_move, ignore_index=True)
        res_df = res_df.append(dict_int_pred, ignore_index=True)
        res_df = res_df.append(dict_ext_pred, ignore_index=True)
        res_df = res_df.append(dict_otherwise, ignore_index=True)
        
        s = f"""

        * Количество глаголов, связанных с чувствами: {counter["Чувства"]}
        * Список глаголов, связанных с чувствами: {list_feel_from_text}
        
        * Количество глаголов, связанных с мыслями: {counter["Мысли"]}
        * Список глаголов, связанных с мыслями: {list_mind_from_text}
        
        * Количество глаголов, связанных с отношением: {counter["Отношение"]}
        * Список глаголов, связанных с отношением: {list_relation_from_text}
        
        * Количество глаголов, связанных с модальностью: {counter["Модальные"]}
        * Список глаголов, связанных с модальностью: {list_modal_from_text}
        
        * Количество глаголов, связанных с умениями: {counter["Умения"]}
        * Список глаголов, связанных с умениями: {list_skills_from_text}
        
        * Количество внутренних предикатов: {counter["Внутренний предикат"]}
        * Внутренние предикаты (глаголы): {list_of_internal_predicat}
        
        * Количество глаголов, связанных с действием: {counter["Действие"]}
        * Список глаголов, связанных с действием: {list_action_from_text}
        
        * Количество глаголов, связанных с коммуникацией: {counter["Коммуникация"]}
        * Список глаголов, связанных с коммуникацией: {list_communication_from_text}
        
        * Количество глаголов, связанных с бытием: {counter["Бытие"]}
        * Список глаголов, связанных с бытием: {list_being_from_text}
        
        * Количество глаголов, связанных с движением: {counter["Движение"]}
        * Список глаголов, связанных с движением: {list_move_from_text}
        
        * Количество внешних предикатов: {counter["Внешний предикат"]}
        * Внешние предикаты (глаголы): {list_of_external_predicat}
        
        * Количество других глаголов: {counter["Другие глаголы"]}
        * Список других глаголов: {list_of_otherwise_verbs}

        """
        color_text = self.color_internal_and_external_predicate_in_text(text, public_semantic_role_df)
        print(color_text)
        print(s)
        cols = ["Категория", "Количество в тексте", "Слова"]
        return res_df[cols]

    def list_of_internal_predicat(self, text):
        dict_with_norm = self.dict_verbs_norm(text)
        verbs_from_text = list(dict_with_norm.keys())
        count = len(verbs_from_text)
        if count == 0:
            count += 1
        list_feel_verbs, list_mind_verbs, list_relation_verbs, list_modal_verbs, list_skills_verbs = verbs_df_to_internal_pred_verbs(self.verbs_df)
        global_list_of_internal_lower = list_feel_verbs + list_mind_verbs + list_relation_verbs + list_modal_verbs + list_skills_verbs
        doc = self.nlp_spacy(text)
        list_of_pattern = []
        list_of_pattern_i = []
        for token in doc:
            if token.text in verbs_from_text:
                if dict_with_norm.get(token.text) in global_list_of_internal_lower:
                    list_of_colloc = []
                    list_of_index = []
                    list_of_colloc.append(token.text)
                    list_of_index.append(token.i)
                    list_of_pattern.append(tuple(list_of_colloc))
                    list_of_pattern_i.append(tuple(list_of_index))
        return list_of_pattern, list_of_pattern_i, count

    def list_of_external_predicat(self, text):
        dict_with_norm = self.dict_verbs_norm(text)
        verbs_from_text = list(dict_with_norm.keys())
        count = len(verbs_from_text)
        if count == 0:
            count += 1
        #text = TextProcess
        list_action_verbs, list_communication_verbs, list_being_verbs, list_move_verbs = verbs_df_to_external_pred_verbs(self.verbs_df)
        global_list_of_external_lower = list_action_verbs + list_communication_verbs + list_being_verbs + list_move_verbs
        doc = self.nlp_spacy(text)
        list_of_pattern = []
        list_of_pattern_i = []
        for token in doc:
            if token.text in verbs_from_text:
                if dict_with_norm.get(token.text) in global_list_of_external_lower:
                    list_of_colloc = []
                    list_of_index = []
                    list_of_colloc.append(token.text)
                    list_of_index.append(token.i)
                    list_of_pattern.append(tuple(list_of_colloc))
                    list_of_pattern_i.append(tuple(list_of_index))
        return list_of_pattern, list_of_pattern_i, count



    def color_text_by_list_of_ind(self, text, list_of_ind, color):
        doc = self.nlp_spacy(text)
        color_text = ""
        for token in doc:
            if TextProcessor.is_in_any(token.i, list_of_ind):
                color_text += colored(token.text, color)
                color_text += " "
            else:
                color_text += token.text
                color_text += " "
        res_text = TextProcessor.return_the_style_of_text(color_text)
        return res_text

    def color_text_by_2lists_of_ind(self, text, list_of_ind1, list_of_ind2, color1, color2):
        doc = self.nlp_spacy(text)
        color_text = ""
        for token in doc:
            if TextProcessor.is_in_any(token.i, list_of_ind1):
                color_text += colored(token.text, color1)
                color_text += " "
            elif TextProcessor.is_in_any(token.i, list_of_ind2):
                color_text += colored(token.text, color2)
                color_text += " "
            else:
                color_text += token.text
                color_text += " "
        res_text = TextProcessor.return_the_style_of_text(color_text)
        return res_text

    @staticmethod
    def return_the_style_of_text(text):
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" / ", "/")
        step6 = step5.replace(" ` ", " '").replace(" .", ".").replace(" ,", ",").strip()
        return step6
        
        
    @staticmethod
    def dict_with_text_indicators_events(glue_text, person_public_df):
        dict_res = {}
        dict_res['Доля гласных / Доля согласных'] = person_public_df[person_public_df["Текст"] == glue_text]["Гласные, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Согласные, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Согласные, %"].values[0] != 0 else 1)
        dict_res['Несовершенный вид / Совершенный вид'] = person_public_df[person_public_df["Текст"] == glue_text]["Несовершенный вид, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Совершенный вид, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Совершенный вид, %"].values[0] != 0 else 1)
        dict_res['Количество глаголов / Количество прилагательных'] = person_public_df[person_public_df["Текст"] == glue_text]["Глаголы, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Прилагательные, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Прилагательные, %"].values[0] != 0 else 1)
        dict_res['Количество наречий / Количество деепричастий'] = person_public_df[person_public_df["Текст"] == glue_text]["Наречия, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0] != 0 else 1)
        dict_res['Количество причастий / Количество деепричастий'] = person_public_df[person_public_df["Текст"] == glue_text]["Причастия, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Деепричастия, %"].values[0] != 0 else 1)
        dict_res['Доля условного наклонения / Доля повелительного наклонения'] = person_public_df[person_public_df["Текст"] == glue_text]["Условное наклонение, %"].values[0] / (person_public_df[person_public_df["Текст"] == glue_text]["Повелительное наклонение, %"].values[0] if person_public_df[person_public_df["Текст"] == glue_text]["Повелительное наклонение, %"].values[0] != 0 else 1)
        dict_res['Доля частиц НЕ среди всех слов'] = person_public_df[person_public_df["Текст"] == glue_text]["Доля частиц НЕ"].values[0] /             person_public_df[person_public_df["Текст"] == glue_text]["Общее количество слов в тексте"].values[0]
        return dict_res
        
    @staticmethod
    def indicator_little(text, feature_name, person_public_df, public_df):
        feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
        q_33 = public_df[feature_name].quantile(q=0.33)
        if feature_value <= q_33:
            return 1
        return 0

    @staticmethod
    def indicator_mean(text, feature_name, person_public_df, public_df):
        feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
        q_33 = public_df[feature_name].quantile(q=0.33)
        q_66 = public_df[feature_name].quantile(q=0.66)
        if feature_value <= q_66 and feature_value > q_33:
            return 1
        return 0

    @staticmethod
    def indicator_large(text, feature_name, person_public_df, public_df):
        feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
        q_66 = public_df[feature_name].quantile(q=0.66)
        if feature_value > q_66:
            return 1
        return 0
        
        
    def table_random_syst_by_token(self, text, person_public_df, df_for_rand_syst_indicators):
        dict_with_indicators_events = self.dict_with_text_indicators_events(text, person_public_df)
        dict_with_indicators_events["Текст"] = text
        dict_for_df = {k[0]: [k[1]] for k in dict_with_indicators_events.items()}
        df_temp = pd.DataFrame.from_dict(dict_for_df)
        person_df = pd.DataFrame()
        person_df = person_df.append(df_temp, ignore_index=True)

        list_of_inverse_quantile = []
        list_of_feature_name = []
        list_of_ind_value = []
        dict_temp = {}
        
        feature_name = 'Доля гласных / Доля согласных'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)
            
        feature_name = 'Несовершенный вид / Совершенный вид'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)
            
        feature_name = 'Количество глаголов / Количество прилагательных'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_little(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)  
            
        feature_name = 'Количество наречий / Количество деепричастий'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)
            
        feature_name = 'Количество причастий / Количество деепричастий'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)
            
        feature_name = 'Доля условного наклонения / Доля повелительного наклонения'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(0)
        elif self.indicator_mean(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(1)
        else:
            list_of_ind_value.append(2)
            
        feature_name = 'Доля частиц НЕ среди всех слов'
        list_of_feature_name.append(feature_name)
        feature_value = self.feature_value_by_text(text, feature_name, person_df)
        list_of_inverse_quantile.append(self.inverse_percentile(df_for_rand_syst_indicators[feature_name].to_list(), feature_value))
        if self.indicator_large(text, feature_name, person_df, df_for_rand_syst_indicators):
            list_of_ind_value.append(2)
        else:
            list_of_ind_value.append(1)
            
            
        feature_name = 'Хаотичная/Системная'
        list_of_feature_name.append(feature_name)
        counter = Counter(list_of_ind_value)
        if counter[2] > counter[0]:
            feature_value = 2
        elif counter[0] > counter[2]:
            feature_value = 0
        else:
            feature_value = 1
        list_of_inverse_quantile.append(-1)
        list_of_ind_value.append(feature_value)
        
        
        dict_temp = {"Индикатор": list_of_feature_name, "Значение": list_of_ind_value, "Обратный квантиль": np.round(list_of_inverse_quantile, 2)}
        df = pd.DataFrame(dict_temp)
        
        return df
        
    @staticmethod
    def feature_value_by_text(text, feature_name, person_df):
        feature_value = person_df[person_df["Текст"] == text][feature_name].values[0]
        return feature_value

    @staticmethod
    def inverse_percentile(arr, num):
        arr = sorted(arr)
        i_arr = [i for i, x in enumerate(arr) if x > num]
        return i_arr[0] / len(arr) if len(i_arr) > 0 else 1
    
    @staticmethod
    def df_head_of_dict_freq(dict_freq, head_size = 5):
        df = pd.DataFrame(dict_freq, index=['count']).T.head(head_size)
        df.index.name = 'word'
        return df

    def table_by_text_temp_cosy(self, text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
        for line in w.wrap(self.color_part_of_speech_in_text(text)): 
            print(line)
        table = df_with_features_ranges_and_temp_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df)
        display(table)
