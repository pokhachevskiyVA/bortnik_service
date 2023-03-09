#from IPython import get_ipython

import sys
sys.path.append("../../")




import warnings
import logging
import pandas as pd
import json
import sys
from morpholog import Morpholog
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from matplotlib import rcParams
from pymystem3 import Mystem
from simple_elmo import ElmoModel
import stanza
import spacy
import snowballstemmer
import os

logging.disable(sys.maxsize)
warnings.filterwarnings("ignore")




from analytics_lib.nlp_texts.text import TextProcessor
from analytics_lib.nlp_texts.psychotype import *


PATH_RECOURCES = 'analytics_lib/notebooks/'
mystem = Mystem()
nlp_core = stanza.Pipeline('ru', use_gpu=False, dir=f'{PATH_RECOURCES}stanza_resources')
morpholog = Morpholog()
tokenizer = RegexTokenizer()
ftsnm = FastTextSocialNetworkModel(tokenizer=tokenizer)
nlp_spacy = spacy.load("ru_core_news_sm")
stemmer = snowballstemmer.stemmer('russian')

import tensorflow.compat.v1 as tf
tf.reset_default_graph()
elmo_model = ElmoModel()
elmo_model.load(f'{PATH_RECOURCES}elmo_resources')

df_sense = pd.read_pickle("analytics_lib/data/df_sense.pkl")
verbs_df = pd.read_pickle("analytics_lib/data/verbs_df.pkl")




text_processor = TextProcessor(
    m=mystem,
    nlp_core=nlp_core,
    morpholog=morpholog,
    fastTextSocialNetworkModel=ftsnm,
    nlp_spacy=nlp_spacy,
    stemmer = stemmer,
    elmo_model = elmo_model,
    df_sense = df_sense,
    verbs_df = verbs_df
)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
## usage:
# with open('json_example.json', 'w', encoding='utf-8') as f:
#     json.dump(dict_res, f, ensure_ascii=False, cls=NpEncoder, indent=10)


def api(text: str, quantiles: str):
    
    dict_res = {}
    dict_res['Текст'] = text
    dict_res['Тип квантилей'] = quantiles
    
    if quantiles == 'assessty_short':
        PATH = 'analytics_lib/data/'  
    elif quantiles == 'assessty_all':
        PATH = 'analytics_lib/data/update_dataframes/'
    elif quantiles == 'dialogs':  
        PATH = 'analytics_lib/data/telecom/'
    else:
        raise RuntimeError('Неверное название средних')

    public_df = pd.read_pickle(f"{PATH}public_df.pkl")
    public_semantic_role_df = pd.read_pickle(f"{PATH}public_semantic_role_df.pkl")
    morph_df = pd.read_pickle(f"{PATH}morph_df.pkl")
    df_for_rand_syst_indicators = pd.read_pickle(f"{PATH}df_for_rand_syst_indicators.pkl")
    public_df_for_new_labeling = pd.read_pickle(f"{PATH}df_for_new_labeling.pkl")
    public_modality_df = pd.read_pickle(f"{PATH}public_modality_df.pkl")


    dict_res['Интервалы для темпераментов'] = df_with_sample_ranges_and_temp(public_df, public_semantic_role_df).set_index('Название характеристики').to_dict('index')




    dict_res['Интервалы для психотипов'] = df_with_sample_ranges_and_psych(public_df, public_semantic_role_df, morph_df).set_index('Название характеристики').to_dict('index')

    person_public_df = text_processor.compute_one_df(text, "public_df")
    person_public_semantic_role_df = text_processor.compute_one_df(text, "public_semantic_role_df")
    person_morph_df = text_processor.compute_one_df(text, "morph_df")
    person_public_dict_with_distances = text_processor.compute_one_df(text, "conceptual_dict")
    person_public_modality_df = text_processor.compute_one_df(text, "public_modality_df")
    person_df_for_new_labeling = text_processor.df_of_relative_features(text, person_public_df, person_public_semantic_role_df, person_public_modality_df)


    dict_res['Морфология, синтаксис, семантика'] = person_public_df.to_dict('index')[0]

    dict_res['Времена глаголов'] = person_morph_df.to_dict('index')[0]

    dict_res['Агенсность, Предикаты'] = person_public_semantic_role_df.to_dict('index')[0]

    dict_res['Модальности'] = person_public_modality_df.to_dict('index')[0]

    df_new_labeling = text_processor.df_with_new_labels_by_text(text, person_df_for_new_labeling, public_df_for_new_labeling)
    dict_res['Разметка шкал \"Локльная/Глобальная\" и \"Хаотичная/Системная\"'] = df_new_labeling.to_dict('index')[0]

    dict_res['Индикаторы психотипов в разрезе признаков по тексту'] = df_with_features_ranges_psyh_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df).set_index('Название характеристики').to_dict('index')

    dict_res['Индикаторы психотипов в разрезе признаков по тексту (альтернативный формат)'] = df_with_features_ranges_and_psych_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df).set_index('Название характеристики').to_dict('index')

    dict_psychotype_by_text = dict_with_psychotype_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df)
    dict_res["Психотипы"] = {}
    dict_res["Психотипы"]["Абсолютные значения"] = dict(dict_psychotype_by_text)
    dict_res["Психотипы"]["Нормировка по l1-норме"] = dict_norm(dict_psychotype_by_text)
    dict_res["Психотипы"]["Нормировка по Чебышевской норме"] = dict_norm_cheb(dict_psychotype_by_text)

    dict_count = {}
    dict_group_mean = {}
    #person_public_dict_with_distances = text_processor.compute_one_df(text, "conceptual_dict")
    dict_res['Индекс близости к 4 психотипам'] = person_public_dict_with_distances[text]
    for psych in person_public_dict_with_distances[text].keys():
        dict_count[psych] = person_public_dict_with_distances[text][psych]["count"]
        dict_group_mean[psych] = person_public_dict_with_distances[text][psych]["group_mean"]

    dict_count_res = {k[0]: [k[1]] for k in dict_count.items()}
    dict_group_mean_res = {k[0]: [k[1]] for k in dict_group_mean.items()}

    dict_res["Психотипы по ELMO"] = {}
    dict_res["Психотипы по ELMO"]["count"] = dict_count_res
    dict_res["Психотипы по ELMO"]["dict_group_mean"] = dict_group_mean_res

    dict_res['Индикаторы темпераментов в разрезе признаков по тексту'] = df_with_features_ranges_temp_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df).set_index('Название характеристики').to_dict('index')

    dict_res['Индикаторы темпераментов в разрезе признаков по тексту (альтернативный формат)'] = df_with_features_ranges_and_temp_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df).set_index('Название характеристики').to_dict('index')

    dict_temp_by_text = dict_with_temp_by_text(text, person_public_df, person_public_semantic_role_df,\
                                               public_df, public_semantic_role_df)
    dict_res["Темпераменты"] = {}
    dict_res["Темпераменты"]["Абсолютные значения"] = dict(dict_temp_by_text)
    dict_res["Темпераменты"]["Нормировка по l1-норме"] = dict_norm(dict_temp_by_text)
    dict_res["Темпераменты"]["Нормировка по Чебышевской норме"] = dict_norm_cheb(dict_temp_by_text)

    dict_res["Индикаторы шкалы \"Хаотичная/Системная\" с обратными квантилями"] = text_processor.table_random_syst_by_token(text, person_public_df, df_for_rand_syst_indicators).set_index('Индикатор').to_dict('index')

    return dict_res