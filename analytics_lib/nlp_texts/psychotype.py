import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.spatial import distance
from IPython.display import display, Markdown

def indicator_no(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    minimum = np.min(public_df[feature_name])
    if feature_value <= minimum:
        return 1
    return 0

def indicator_little(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    minimum = np.min(public_df[feature_name])
    q_20 = public_df[feature_name].quantile(q=0.2)
    if feature_value <= q_20 and feature_value > minimum:
        return 1
    return 0

def indicator_less_mean(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    q_20 = public_df[feature_name].quantile(q=0.2)
    q_40 = public_df[feature_name].quantile(q=0.4)
    if feature_value <= q_40 and feature_value > q_20:
        return 1
    return 0

def indicator_mean(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    q_40 = public_df[feature_name].quantile(q=0.4)
    q_60 = public_df[feature_name].quantile(q=0.6)
    if feature_value <= q_60 and feature_value > q_40:
        return 1
    return 0

def indicator_greater_mean(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    q_60 = public_df[feature_name].quantile(q=0.6)
    q_80 = public_df[feature_name].quantile(q=0.8)
    if feature_value <= q_80 and feature_value > q_60:
        return 1
    return 0

def indicator_large(text, feature_name, person_public_df, public_df):
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    q_80 = public_df[feature_name].quantile(q=0.8)
    if feature_value > q_80:
        return 1
    return 0







def get_key_with_max_value(dict_input):
    return max(dict_input, key=dict_input.get)

def dict_with_psychotype_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df):
    df = df_with_features_ranges_psyh_by_text(text, person_public_df, person_public_semantic_role_df, pd.DataFrame(), public_df, public_semantic_role_df, pd.DataFrame())
    dict_res = Counter(np.concatenate(df['Психотип']))
    return dict_res

        
def psychotype_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df):
    print(text)
    dict1 = dict_with_psychotype_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df)
    dict_res = {k[0]: [k[1]] for k in dict1.items()}
    dict1_norm = dict_norm(dict1)
    dict_res_norm = {k[0]: [k[1]] for k in dict1_norm.items()}
    display(Markdown("## * __Значение = Количество активировавшихся индикаторов психотипа__"))
    display(pd.DataFrame.from_dict(dict_res))
    display(Markdown("## * __Значение = Нормированный по l1-норме предыдущий вектор__"))
    display(pd.DataFrame.from_dict(dict_res_norm))
    print("\n")

def df_with_features_and_ranges(public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(public_df._get_numeric_data().columns.values)

    list_of_morph_features = list(morph_df._get_numeric_data().columns.values)

    list_of_semantic_role_features = list(public_semantic_role_df._get_numeric_data().columns.values)
    for feature_name in list_of_text_features:
        dict_text = dict_with_ranges_by_feature_text(feature_name, public_df)
        res_df = res_df.append(dict_text, ignore_index=True)
    for feature_name in list_of_semantic_role_features:
        dict_semantic = dict_with_ranges_by_feature_text(feature_name, public_semantic_role_df)
        res_df = res_df.append(dict_semantic, ignore_index=True)
    for feature_name in list_of_morph_features:
        dict_morph = dict_with_ranges_by_feature_text(feature_name, morph_df)
        res_df = res_df.append(dict_morph, ignore_index=True)
    cols = ["Название характеристики", "Нет", "Мало", "Меньше среднего", "Среднее", "Больше среднего", "Много"]
    res_df = res_df[cols]
    return res_df


def dict_with_ranges_by_feature_text(feature_name, public_df):
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    res_dict["Нет"] = range_no(feature_name, public_df)
    res_dict["Мало"] = range_little(feature_name, public_df)
    res_dict["Меньше среднего"] = range_less_mean(feature_name, public_df)
    res_dict["Среднее"] = range_mean(feature_name, public_df)
    res_dict["Больше среднего"] = range_greater_mean(feature_name, public_df)
    res_dict["Много"] = range_large(feature_name, public_df)
    return res_dict

def dict_with_ranges_by_feature_semantic(feature_name, public_semantic_role_df):
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    res_dict["Нет"] = range_no(feature_name, public_semantic_role_df)
    res_dict["Мало"] = range_little(feature_name, public_semantic_role_df)
    res_dict["Меньше среднего"] = range_less_mean(feature_name, public_semantic_role_df)
    res_dict["Среднее"] = range_mean(feature_name, public_semantic_role_df)
    res_dict["Больше среднего"] = range_greater_mean(feature_name, public_semantic_role_df)
    res_dict["Много"] = range_large(feature_name, public_semantic_role_df)
    return res_dict

def range_no(feature_name, public_df):
    minimum = round(np.min(public_df[feature_name]), 2)
    return tuple([0, minimum])

def range_little(feature_name, public_df):
    minimum = round(np.min(public_df[feature_name]), 2)
    q_20 = round(public_df[feature_name].quantile(q=0.2), 2)
    return tuple([minimum, q_20])

def range_less_mean(feature_name, public_df):
    q_20 = round(public_df[feature_name].quantile(q=0.2), 2)
    q_40 = round(public_df[feature_name].quantile(q=0.4), 2)
    return tuple([q_20, q_40])

def range_mean(feature_name, public_df):
    q_40 = round(public_df[feature_name].quantile(q=0.4), 2)
    q_60 = round(public_df[feature_name].quantile(q=0.6), 2)
    return tuple([q_40, q_60])

def range_greater_mean(feature_name, public_df):
    q_60 = round(public_df[feature_name].quantile(q=0.6), 2)
    q_80 = round(public_df[feature_name].quantile(q=0.8), 2)
    return tuple([q_60, q_80])

def range_large(feature_name, public_df):
    q_80 = round(public_df[feature_name].quantile(q=0.8), 2)
    maximum = round(np.max(public_df[feature_name]), 2)
    return tuple([q_80, maximum])

def df_with_features_ranges_psyh_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(person_public_df._get_numeric_data().columns.values)
    list_of_semantic_role_features = list(person_public_semantic_role_df._get_numeric_data().columns.values)
    #list_of_semantic_role_features = ['Агенс', 'Пациенс', 'Внутренний предикат', 'Внешний предикат']
    #list_of_morph_features = ["Прошедшее, %", "Настоящее, %", "Будущее, %", "Абсолютное, %"]
    for feature_name in list_of_text_features:
        dict_text = dict_with_feature_range_psyh(text, feature_name, person_public_df, public_df)
        res_df = res_df.append(dict_text, ignore_index=True)
    for feature_name in list_of_semantic_role_features:
        dict_semantic = dict_with_feature_range_psyh(text, feature_name, person_public_semantic_role_df, public_semantic_role_df)
        res_df = res_df.append(dict_semantic, ignore_index=True)
    # for feature_name in list_of_morph_features:
    #     dict_morph = dict_with_feature_range_psyh(text, feature_name, person_morph_df, morph_df)
    #     res_df = res_df.append(dict_morph, ignore_index=True)
    cols = ["Название характеристики", "Значение", "Количество", "Диапазон", "Психотип"]
    res_df = res_df[cols]
    return res_df

def dict_with_feature_range_psyh(text, feature_name, person_public_df, public_df):
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    res_dict["Значение"] = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    size, rang, psyh = size_range_psyh_of_feature_value(text, feature_name, person_public_df, public_df)
    res_dict["Количество"] = size
    res_dict["Диапазон"] = rang
    res_dict["Психотип"] = psyh
    return res_dict



def size_range_psyh_of_feature_value(text, feature_name, person_public_df, public_df):
    size = ""
    rang = tuple()
    psyh = []
    if indicator_no(text, feature_name, person_public_df, public_df) == 1:
        size = "Нет"
        rang = range_no(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
        
    elif indicator_little(text, feature_name, person_public_df, public_df) == 1:
        size = "Мало"
        rang = range_little(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
        
    elif indicator_less_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Меньше среднего"
        rang = range_less_mean(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
        
    elif indicator_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Среднее"
        rang = range_mean(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
        
    elif indicator_greater_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Больше среднего"
        rang = range_greater_mean(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
        
    elif indicator_large(text, feature_name, person_public_df, public_df) == 1:
        size = "Много"
        rang = range_large(feature_name, public_df)
        psyh = psyh_by_feature_name_and_size(feature_name, size)
    return size, rang, psyh

def psyh_by_feature_name_and_size(feature_name, size):
    list_with_psyh = []
    if feature_name == 'Доля коротких предложений':
        if size == "Среднее":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Эмотив")
        elif size == "Много":
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Шизоид")
            
    elif feature_name == 'Доля длинных предложений':
        if size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Нет":
            list_with_psyh.append("Гипертим")
        elif size == "Много":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Тревожный")
        elif size == "Мало":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Эмотив")
            
    elif feature_name == 'Доля длинных низкочастотных слов':
        if size == "Больше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Нет":
            list_with_psyh.append("Эпилептоид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Тревожный")
        elif size == "Мало":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Истероид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Гипертим")
        
    elif feature_name == 'Доля коротких высокочастотных слов':
        if size == "Много":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Тревожный")
        elif size == "Среднее":
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Параноял")
            
    elif feature_name == 'Доля восклицательных предложений':
        if size == "Много":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эмотив")
            
    elif feature_name == 'Доля вопросительных предложений':
        if size == "Много":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Тревожный")
            
    elif feature_name == 'Доля частиц НЕ':
        if size == "Много":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Депрессивный")
        elif size == "Мало":
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Гипертим")
        elif size == "Среднее":
            list_with_psyh.append("Истероид")
            
    elif feature_name == 'Гласные, %':
        if size == "Много":
            list_with_psyh.append("Тревожный")
        elif size == "Мало":
            list_with_psyh.append("Параноял")
        elif size == "Среднее":
            list_with_psyh.append("Гипертим")
        elif size == "Больше среднего":
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Истероид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Шизоид")
            
    elif feature_name == 'Согласные, %':
        if size == "Много":
            list_with_psyh.append("Параноял")
        elif size == "Мало":
            list_with_psyh.append("Тревожный")
        elif size == "Среднее":
            list_with_psyh.append("Гипертим")
        elif size == "Больше среднего":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Эмотив")
            
    elif feature_name == 'Существительные, %':
        if size == "Много":
            list_with_psyh.append("Эпилептоид")
            
    elif feature_name == 'Глаголы, %':
        if size == "Много":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Эмотив")
        elif size == "Меньше среднего":
            list_with_psyh.append("Истероид")
            
    elif feature_name == 'Прилагательные, %':
        if size == "Много":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Эмотив")
        elif size == "Мало":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эпилептоид")
            
    elif feature_name == 'Наречия, %':
        if size == "Много":
            list_with_psyh.append("Истероид")
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Эмотив")
        elif size == "Меньше среднего":
            list_with_psyh.append("Гипертим")
        elif size == "Больше среднего":
            list_with_psyh.append("Шизоид")
            
    elif feature_name == 'Причастия, %':
        if size == "Много":
            list_with_psyh.append("Шизоид")
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Эмотив")
        elif size == "Меньше среднего":
            list_with_psyh.append("Гипертим")
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
            
    elif feature_name == 'Деепричастия, %':
        if size == "Много":
            list_with_psyh.append("Шизоид")
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Эмотив")
        elif size == "Меньше среднего":
            list_with_psyh.append("Гипертим")
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
            
    elif feature_name == 'Повелительное наклонение, %':
        if size == "Много":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Тревожный")
            
    elif feature_name == 'Агенс':
        if size == "Много":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Истероид")
        elif size == "Мало":
            list_with_psyh.append("Депрессивный")
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Эпилептоид")
        elif size == "Больше среднего":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Шизоид")
            
    elif feature_name == 'Пациенс':
        if size == "Много":
            list_with_psyh.append("Депрессивный")
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Эпилептоид")
        elif size == "Мало":
            list_with_psyh.append("Параноял")
        elif size == "Меньше среднего":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Истероид")
        elif size == "Среднее":
            list_with_psyh.append("Шизоид")
            
    elif feature_name == 'Внутренний предикат':
        if size == "Много":
            list_with_psyh.append("Эмотив")
        elif size == "Мало":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Истероид")
        elif size == "Больше среднего":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Шизоид")
        elif size == "Среднее":
            list_with_psyh.append("Эпилептоид")
            
    elif feature_name == 'Внешний предикат':
        if size == "Много":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Истероид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Шизоид")
        elif size == "Больше среднего":
            list_with_psyh.append("Тревожный")
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            list_with_psyh.append("Эпилептоид")

    elif feature_name == 'Доля предложений на одно слово':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Средняя длина предложения':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Cредняя длинна слова в тексте':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Среднее число запятых в предложении':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Гласная "И", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Гласная "О", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Эмотив")

    elif feature_name == 'Гласная "А", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Эпилептоид")

    elif feature_name == 'Совершенный вид, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Шизоид")
        elif size == "Много":
            list_with_psyh.append("Эпилептоид")

    elif feature_name == 'Несовершенный вид, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Шизоид")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Эмотив")
        elif size == "Много":
            pass

    elif feature_name == 'Превосходная степень, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Истероид")
        elif size == "Много":
            pass

    elif feature_name == 'Сравнительная степень, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Эмотив")
        elif size == "Много":
            pass

    elif feature_name == 'Изъявительное наклонение, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Гипертим")
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Тревожный")

    elif feature_name == 'Условное наклонение, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            list_with_psyh.append("Истероид")
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Эмотив")

    elif feature_name == 'Среднее количество слов в предложении':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Средняя доля именованных сущностей в предложении':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Эпилептоид")
            list_with_psyh.append("Параноял")
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Тревожный")
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Эмотив")
            list_with_psyh.append("Истероид")

    # elif feature_name == 'Средняя доля именованных сущностей в предложении':
    #     if size == "Нет":
    #         pass
    #     elif size == "Мало":
    #         pass
    #     elif size == "Меньше среднего":
    #         pass
    #     elif size == "Среднее":
    #         pass
    #     elif size == "Больше среднего":
    #         pass
    #     elif size == "Много":
    #         pass

    elif feature_name == 'Среднее количество морфем в слове':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Средняя частота TOP10 самых редких слов в тексте':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            list_with_psyh.append("Эмотив")
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            list_with_psyh.append("Истероид")
        elif size == "Много":
            list_with_psyh.append("Шизоид")

    elif feature_name == 'Среднее число синтаксических веток (по предложениям)':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Среднее число синтаксических веток (на одно слово)':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Средняя глубина синтаксической ветки':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Среднее число прилагательных в синтаксической ветке':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Среднее число причастий в синтаксической ветке':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Среднее число деепричастий в синтаксической ветке':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == "Прошедшее, %":
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == "Настоящее, %":
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == "Будущее, %":
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == "Абсолютное, %":
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Доля сложноподчиненных предложений':
        if size == "Нет":
            pass
        elif size == "Мало":
            list_with_psyh.append("Истероид")
            list_with_psyh.append("Гипертим")
            list_with_psyh.append("Эпилептоид")
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            list_with_psyh.append("Параноял")
            list_with_psyh.append("Эмотив")
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Шизоид")
            list_with_psyh.append("Тревожный")
            
    return list_with_psyh

def image_of_ranges():
    dict_range = dict()
    dict_range["Нет"] = ("0", "minimum")
    dict_range["Очень мало, Мало"] = ("minimum", "q_20")
    dict_range["Чуть меньше среднего, Меньше среднего"] = ("q_20", "q_40")
    dict_range["Среднее"] = ("q_40", "q_60")
    dict_range["Чуть больше среднего, Больше среднего"] = ("q_60", "q_80")
    dict_range["Много, Очень много"] = ("q_80", "inf")
    res_df = pd.DataFrame()
    res_df = res_df.append(dict_range, ignore_index=True)
    cols = ["Нет", "Очень мало, Мало", "Чуть меньше среднего, Меньше среднего", "Среднее", "Чуть больше среднего, Больше среднего", "Много, Очень много"]
    res_df = res_df[cols]
    return res_df

def dict_with_sample_ranges_and_psych(feature_name, public_df, public_semantic_role_df, morph_df):
    list_of_all_types = ["Истероид", "Гипертим", "Шизоид", "Параноял", "Эпилептоид", "Депрессивный", "Эмотив", "Тревожный"]
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    for psyh_name in list_of_all_types:
        res_dict[psyh_name] = range_with_size_by_psyh_and_feat_name(feature_name, psyh_name, public_df, public_semantic_role_df, morph_df)
    return res_dict

def df_with_sample_ranges_and_psych(public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(public_df._get_numeric_data().columns.values) + \
                                                                list(morph_df._get_numeric_data().columns.values)
    for feature_name in list_of_text_features:
        dict_feat = dict_with_sample_ranges_and_psych(feature_name, public_df, public_semantic_role_df, morph_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    cols = ["Название характеристики", "Истероид", "Гипертим", "Шизоид", "Параноял", "Эпилептоид", "Депрессивный", "Эмотив", "Тревожный"]
    res_df = res_df[cols]
    return res_df

def range_with_size_by_psyh_and_feat_name(feature_name, psyh_name, public_df, public_semantic_role_df, morph_df):
    if feature_name in list(public_df.columns.values):
        for size, func in zip(["Мало", "Меньше среднего", "Среднее", "Больше среднего", "Много"],
                       [range_little, range_less_mean, range_mean, range_greater_mean, range_large]):
            if psyh_name in psyh_by_feature_name_and_size(feature_name, size):
                return f"{size}: {func(feature_name, public_df)}"
    elif feature_name in list(public_semantic_role_df.columns.values):
        for size, func in zip(["Мало", "Меньше среднего", "Среднее", "Больше среднего", "Много"],
                   [range_little, range_less_mean, range_mean, range_greater_mean, range_large]):
            if psyh_name in psyh_by_feature_name_and_size(feature_name, size):
                return f"{size}: {func(feature_name, public_semantic_role_df)}"
    return "-"

def dict_with_feature_values_range_psyh_by_text(text, feature_name, person_public_df, public_df):
    res_dict = {}
    size, rang, psyh = size_range_psyh_of_feature_value(text, feature_name, person_public_df, public_df)
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    list_of_all_types = ["Истероид", "Гипертим", "Шизоид", "Параноял", "Эпилептоид", "Депрессивный", "Эмотив", "Тревожный"]
    res_dict["Название характеристики"] = feature_name
    for psyh_type in psyh:
        res_dict[psyh_type] = f"{size}: {feature_value} из {rang}"
    for psyh_type in list_of_all_types:
        if psyh_type not in psyh:
            res_dict[psyh_type] = "-"
    return res_dict

def df_with_features_ranges_and_psych_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(person_public_df._get_numeric_data().columns.values)
    list_of_semantic_role_features = list(person_public_semantic_role_df._get_numeric_data().columns.values)
    list_of_morph_features = list(person_morph_df._get_numeric_data().columns.values)
    for feature_name in list_of_text_features:
        dict_feat = dict_with_feature_values_range_psyh_by_text(text, feature_name, person_public_df, public_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    for feature_name in list_of_semantic_role_features:
        dict_feat = dict_with_feature_values_range_psyh_by_text(text, feature_name, person_public_semantic_role_df, public_semantic_role_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    for feature_name in list_of_morph_features:
        dict_feat = dict_with_feature_values_range_psyh_by_text(text, feature_name, person_morph_df, morph_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    cols = ["Название характеристики", "Истероид", "Гипертим", "Шизоид", "Параноял", "Эпилептоид", "Депрессивный", "Эмотив", "Тревожный"]
    res_df = res_df[cols]
    return res_df

def dict_norm(dict_input):
    sum_of_all = 0
    dict_res = {}
    for key in dict_input.keys():
        sum_of_all += dict_input[key]
    if sum_of_all == 0:
        sum_of_all = 1
    for key in dict_input.keys():
        dict_res[key] = round(dict_input[key] / sum_of_all, 2)
    return dict_res

def dict_norm_cheb(dict_input):
    maximum = np.max(list(dict_input.values()))
    dict_res = {}
    if maximum == 0:
        maximum = 1
    for key in dict_input.keys():
        dict_res[key] = round(dict_input[key] / maximum, 2)
    return dict_res

def text_to_list_of_tokens(text, nlp_core):
    list_of_tokens = []
    processed_text = nlp_core(text)
    for sentence in processed_text.sentences:
        for token in sentence.tokens:
            if not token.words:
                continue
            token = token.words[0]
            if token.upos not in ["PUNCT", "ADP", "CCONJ", "SCONJ", "NUM", "PART", "PRON"]:
                list_of_tokens.append([token.text])
    return list_of_tokens

def distance_between_word_and_psychotypes(emb_word, emb_all_psychotypes_words):
    dict_res = defaultdict()
    dict_res["Истероид"] = distance_between_word_and_list_of_words(emb_word, emb_all_psychotypes_words[0: 9])
    dict_res["Шизоид"] = distance_between_word_and_list_of_words(emb_word, emb_all_psychotypes_words[9: 20])
    dict_res["Эпилептоид"] = distance_between_word_and_list_of_words(emb_word, emb_all_psychotypes_words[20: 29])
    dict_res["Эмотив"] = distance_between_word_and_list_of_words(emb_word, emb_all_psychotypes_words[29: 39])
    return dict_res

def normalize(word_vec):
    norm = np.linalg.norm(word_vec)
    if norm < 1e-5: 
       return word_vec
    return word_vec / norm

def normalize_list(list_word_vec):
    res_list = []
    for word_vec in list_word_vec:
        norm = np.linalg.norm(word_vec)
        if norm < 1e-5:
            res_list.append(word_vec)
        else:
            res_list.append(word_vec / norm)
    return res_list

def distance_between_word_and_list_of_words(emb_word, emb_list_of_words):
    distances = distance.cdist(emb_word, emb_list_of_words, 'cosine')
    return round(np.min(distances),2)

def distance_between_text_and_psychotypes(text, model, nlp_core):
    #res_df = pd.DataFrame()
    res_df = pd.DataFrame(columns=["Слово", "Истероид", "Шизоид", "Эпилептоид", "Эмотив"])
    list_of_tokens = text_to_list_of_tokens(text, nlp_core)
    isteroid_words = [["социальный::статус"], ["свобода"], ["новизна"], ["удовольствие"], ["уникальность"], ["импульсивность"], ["разнообразие"], ["наслаждение::жизнью"], ["новые::впечатления"]]
    shizoid_words = [["достижения"], ["функциональность"], ["индивидуализм"], ["логика"], ["цели"], ["время"], ["эффективность"], ["исследование"], ["независимость"], ["конкуренция"], ["информация"]]
    epileptoid_words = [["выживание"], ["удержание"], ["контроль"], ["накопление"], ["традиции"], ["стабильность"],  ["авторитет"], ["семья"], ["здоровье"]]
    emotiv_words = [["защищенность"], ["принадлежность"], ["миролюбие"], ["спокойствие"], ["полезность"], ["благополучие"], ["альтруизм"], ["идеалы"], ["комфорт"], ["доверие"]]
    all_psychotype_words = isteroid_words + shizoid_words + epileptoid_words + emotiv_words
    emb_all_psychotypes_words = model.get_elmo_vectors(all_psychotype_words, layers = "average")
    emb_all_psychotypes_words = emb_all_psychotypes_words.reshape(39, 1024)
    if not list_of_tokens:
        return res_df
    emb_text = model.get_elmo_vectors(list_of_tokens, layers = "average")
    for emb_word, token in zip(emb_text, list_of_tokens):
        dict_temp = defaultdict()
        dict_temp = distance_between_word_and_psychotypes(emb_word, emb_all_psychotypes_words)
        dict_temp["Слово"] = token[0]
        res_df = res_df.append(dict_temp, ignore_index=True)
    cols = ["Слово", "Истероид", "Шизоид", "Эпилептоид", "Эмотив"]
    return res_df[cols]

def heatmap_df(df):
    return df.style.background_gradient(cmap='RdYlGn_r').set_precision(2)

def processed_df_with_dist(df):
    df_ist = df[df["Истероид"] < 0.62]
    df_shiz = df[df["Шизоид"] < 0.62]
    df_epi = df[df["Эпилептоид"] < 0.62]
    df_emo = df[df["Эмотив"] < 0.62]
    return df_ist, df_shiz, df_epi, df_emo

def text_to_dict_with_dist_mean(text, model, nlp_core):
    dict_res = {}
    df_dist_by_words = distance_between_text_and_psychotypes(text, model, nlp_core)
    df_ist, df_shiz, df_epi, df_emo = processed_df_with_dist(df_dist_by_words)
    df_dist_by_words_processed = pd.concat([df_ist, df_shiz, df_epi, df_emo], ignore_index = True)
    df_mean = np.mean(df_dist_by_words_processed)

    if not np.isnan(df_mean["Истероид"]):
        dict_res["Истероид"] = df_mean["Истероид"]
    else:
        dict_res["Истероид"] = 1

    if not np.isnan(df_mean["Шизоид"]):
        dict_res["Шизоид"] = df_mean["Шизоид"]
    else:
        dict_res["Шизоид"] = 1

    if not np.isnan(df_mean["Эпилептоид"]):
        dict_res["Эпилептоид"] = df_mean["Эпилептоид"]
    else:
        dict_res["Эпилептоид"] = 1

    if not np.isnan(df_mean["Эмотив"]):
        dict_res["Эмотив"] = df_mean["Эмотив"]
    else:
        dict_res["Эмотив"] = 1

    return dict_res

def text_to_dict_with_dist_count(text, model, nlp_core):
    dict_res = {}
    df_dist_by_words = distance_between_text_and_psychotypes(text, model, nlp_core)
    df_ist, df_shiz, df_epi, df_emo = processed_df_with_dist(df_dist_by_words)
    dict_res["Истероид"] = len(df_ist)
    dict_res["Шизоид"] = len(df_shiz)
    dict_res["Эпилептоид"] = len(df_epi)
    dict_res["Эмотив"] = len(df_emo)
    dict_res_norm = dict_norm(dict_res)
    return dict_res_norm

def df_with_all_texts_dist_count(list_of_all_texts, model, nlp_core):
    res_df = pd.DataFrame()
    for text in list_of_all_texts:
        dict_temp = {}
        dict_by_text = text_to_dict_with_dist_count(text, model, nlp_core)
        dict_temp["Текст"] = text
        dict_temp["Истероид"] = dict_by_text["Истероид"]
        dict_temp["Шизоид"] = dict_by_text["Шизоид"]
        dict_temp["Эпилептоид"] = dict_by_text["Эпилептоид"]
        dict_temp["Эмотив"] = dict_by_text["Эмотив"]
        res_df = res_df.append(dict_temp, ignore_index=True)
    cols = ["Текст", "Истероид", "Шизоид", "Эпилептоид", "Эмотив"]
    return res_df[cols]       

def text_to_dict_with_dist_group_mean(text, model, nlp_core):
    dict_res = {}
    df_dist_by_words = distance_between_text_and_psychotypes(text, model, nlp_core)
    df_ist, df_shiz, df_epi, df_emo = processed_df_with_dist(df_dist_by_words)

    if not np.isnan(np.mean(df_ist["Истероид"])):
        dict_res["Истероид"] = np.mean(df_ist["Истероид"])
    else:
        dict_res["Истероид"] = 1

    if not np.isnan(np.mean(df_shiz["Шизоид"])):
        dict_res["Шизоид"] = np.mean(df_shiz["Шизоид"])
    else:
        dict_res["Шизоид"] = 1

    if not np.isnan(np.mean(df_epi["Эпилептоид"])):
        dict_res["Эпилептоид"] = np.mean(df_epi["Эпилептоид"])
    else:
        dict_res["Эпилептоид"] = 1

    if not np.isnan(np.mean(df_emo["Эмотив"])):
        dict_res["Эмотив"] = np.mean(df_emo["Эмотив"])
    else:
        dict_res["Эмотив"] = 1

    return dict_res

def df_with_all_texts_dist_group_mean(list_of_all_texts, model, nlp_core):
    res_df = pd.DataFrame()
    for text in list_of_all_texts:
        dict_temp = {}
        dict_by_text = text_to_dict_with_dist_group_mean(text, model, nlp_core)
        dict_temp["Текст"] = text
        dict_temp["Истероид"] = dict_by_text["Истероид"]
        dict_temp["Шизоид"] = dict_by_text["Шизоид"]
        dict_temp["Эпилептоид"] = dict_by_text["Эпилептоид"]
        dict_temp["Эмотив"] = dict_by_text["Эмотив"]
        res_df = res_df.append(dict_temp, ignore_index=True)
    cols = ["Текст", "Истероид", "Шизоид", "Эпилептоид", "Эмотив"]
    return res_df[cols]

def df_with_all_texts_dist_mean(list_of_all_texts, model, nlp_core):
    res_df = pd.DataFrame()
    for text in list_of_all_texts:
        dict_temp = {}
        dict_by_text = text_to_dict_with_dist_mean(text, model, nlp_core)
        dict_temp["Текст"] = text
        dict_temp["Истероид"] = dict_by_text["Истероид"]
        dict_temp["Шизоид"] = dict_by_text["Шизоид"]
        dict_temp["Эпилептоид"] = dict_by_text["Эпилептоид"]
        dict_temp["Эмотив"] = dict_by_text["Эмотив"]
        res_df = res_df.append(dict_temp, ignore_index=True)
    cols = ["Текст", "Истероид", "Шизоид", "Эпилептоид", "Эмотив"]
    return res_df[cols]

def sum_of_dicts(dict_1, dict_2):
    a_counter = Counter(dict_1)
    b_counter = Counter(dict_2)
    add_dict = a_counter + b_counter
    dict_3 = dict(add_dict)
    return dict_3


def temp_by_feature_name_and_size(feature_name, size):
    list_with_psyh = []
    
    

    
    if feature_name == 'Гласные, %':
        if size == "Больше среднего":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Холерик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Сангвиник")
            
    
    elif feature_name == 'Гласная "И", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Сангвиник")

    elif feature_name == 'Гласная "О", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass

    elif feature_name == 'Гласная "А", %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            list_with_psyh.append("Флегматик")
            
    elif feature_name == 'Согласные, %':
        if size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Существительные, %':
        if size == "Много":
            list_with_psyh.append("Флегматик")
        elif size == "Больше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Среднее":
            list_with_psyh.append("Сангвиник")
        elif size == "Меньше среднего":
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Прилагательные, %':
        if size == "Много":
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Флегматик")
        elif size == "Среднее":
            list_with_psyh.append("Сангвиник")
        elif size == "Меньше среднего":
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Глаголы, %':
        if size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Среднее":
            list_with_psyh.append("Холерик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Флегматик")
            
    elif feature_name == 'Доля местоимений в тексте':
        if size == "Много":
            list_with_psyh.append("Сангвиник")
        elif size == "Больше среднего":
            list_with_psyh.append("Флегматик")
        elif size == "Среднее":
            list_with_psyh.append("Меланхолик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Доля частиц в тексте':
        if size == "Больше среднего":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Сангвиник")
        elif size == "Среднее":
            list_with_psyh.append("Холерик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Флегматик")
            
                
    elif feature_name == 'Наречия, %':
        if size == "Много":
            list_with_psyh.append("Холерик")
        elif size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Среднее":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
            
                
    elif feature_name == 'Причастия, %':
        if size == "Много":
            list_with_psyh.append("Сангвиник")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
            
    elif feature_name == 'Деепричастия, %':
        if size == "Много":
            list_with_psyh.append("Сангвиник")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
            
    elif feature_name == 'Доля союзов в тексте':
        if size == "Много":
            list_with_psyh.append("Флегматик")
        elif size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Мало":
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Доля предлогов в тексте':
        if size == "Много":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
        elif size == "Мало":
            list_with_psyh.append("Сангвиник")
            
    elif feature_name == 'Доля числительных в тексте':
        if size == "Больше среднего":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Холерик")
        elif size == "Мало":
            list_with_psyh.append("Меланхолик")
            
    
    elif feature_name == 'Доля междометий в тексте':
        if size == "Больше среднего":
            list_with_psyh.append("Флегматик")
        elif size == "Мало":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Меланхолик")
        if size == "Среднее":
            list_with_psyh.append("Холерик")
            
    
    elif feature_name == 'Собственные/Нарицательные':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Меланхолик")
            
    elif feature_name == 'Совершенный вид, %':
        if size == "Меньше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Флегматик")

    elif feature_name == 'Несовершенный вид, %':
        if size == "Больше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Меланхолик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Мало":
            list_with_psyh.append("Флегматик")
            
    elif feature_name == 'Превосходная степень, %':
        if size == "Больше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Флегматик")

    elif feature_name == 'Сравнительная степень, %':
        if size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Меланхолик")
            
            
    elif feature_name == 'Среднее количество слов в предложении':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник")
            
    elif feature_name == 'Доля предложений на одно слово':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник")      
            
    elif feature_name == 'Доля длинных предложений':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник") 
         
    elif feature_name == 'Доля коротких предложений':
        if size == "Мало":
            list_with_psyh.append("Сангвиник")
        elif size == "Среднее":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Меланхолик")
        elif size == "Много":
            list_with_psyh.append("Флегматик") 
            
    elif feature_name == 'Среднее число запятых в предложении':
        if size == "Больше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Флегматик")


    elif feature_name == 'Доля сложноподчиненных предложений':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Холерик")
        elif size == "Среднее":
            list_with_psyh.append("Меланхолик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник") 
            
    elif feature_name == 'Доля восклицательных предложений':
        if size == "Мало":
            list_with_psyh.append("Меланхолик")
        elif size == "Среднее":
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Холерик") 
            
    elif feature_name == 'Доля вопросительных предложений':
        if size == "Много":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Меланхолик")
            
    elif feature_name == 'Средняя частота TOP10 самых редких слов в тексте':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник") 
    
    
    elif feature_name == 'Доля повторов в тексте':
        if size == "Мало":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Флегматик")
            
    elif feature_name == 'Вариабельность словаря':
        if size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Меланхолик")
             
            
    elif feature_name == 'Cредняя длинна слова в тексте':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник") 
            
    elif feature_name == 'Доля длинных низкочастотных слов':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Холерик")
        elif size == "Среднее":
            list_with_psyh.append("Меланхолик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник") 
            
    elif feature_name == 'Доля коротких высокочастотных слов':
        if size == "Среднее":
            list_with_psyh.append("Флегматик")
        elif size == "Много":
            list_with_psyh.append("Холерик")
            
    elif feature_name == 'Водность текста':
        if size == "Мало":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Флегматик")
            
    elif feature_name == 'Академическая тошнота':
        if size == "Среднее":
            list_with_psyh.append("Меланхолик")
        elif size == "Много":
            list_with_psyh.append("Сангвиник")
            list_with_psyh.append("Флегматик")
            
            
    elif feature_name == 'Агенс':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
        elif size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Холерик") 

    elif feature_name == 'Пациенс':
        if size == "Меньше среднего":
            list_with_psyh.append("Холерик") 
        elif size == "Среднее":
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Флегматик")
            list_with_psyh.append("Меланхолик")
                       
            
    elif feature_name == 'Внутренний предикат':
        if size == "Мало":
            list_with_psyh.append("Холерик")
        elif size == "Среднее":
            list_with_psyh.append("Флегматик")
        elif size == "Больше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Меланхолик") 
            
            
    elif feature_name == 'Внешний предикат':
        if size == "Среднее":
            list_with_psyh.append("Флегматик")
        elif size == "Меньше среднего":
            list_with_psyh.append("Сангвиник")
        elif size == "Больше среднего":
            list_with_psyh.append("Меланхолик")
        elif size == "Много":
            list_with_psyh.append("Холерик") 
            
            
    elif feature_name == 'Изъявительное наклонение, %':
        if size == "Много":
            list_with_psyh.append("Холерик") 
            list_with_psyh.append("Меланхолик")
            
            
    elif feature_name == 'Условное наклонение, %':
        if size == "Мало":
            list_with_psyh.append("Флегматик")
        elif size == "Среднее":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Меланхолик") 
            
    elif feature_name == 'Повелительное наклонение, %':
        if size == "Мало":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Холерик")
            list_with_psyh.append("Сангвиник")
        elif size == "Много":
            list_with_psyh.append("Флегматик") 
            
    elif feature_name == 'Доля частиц НЕ':
        if size == "Мало":
            list_with_psyh.append("Меланхолик")
        elif size == "Среднее":
            list_with_psyh.append("Холерик")
        elif size == "Много":
            list_with_psyh.append("Флегматик") 
        
    elif feature_name == 'Доля местоимений Я':
        if size == "Много":
            list_with_psyh.append("Меланхолик")
            list_with_psyh.append("Сангвиник")
            
    elif feature_name == 'Настоящее, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass
        
    elif feature_name == 'Будущее, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass
            
    elif feature_name == 'Прошедшее, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass
            
            
    elif feature_name == 'Абсолютное, %':
        if size == "Нет":
            pass
        elif size == "Мало":
            pass
        elif size == "Меньше среднего":
            pass
        elif size == "Среднее":
            pass
        elif size == "Больше среднего":
            pass
        elif size == "Много":
            pass
              
    return list_with_psyh


def size_range_temp_of_feature_value(text, feature_name, person_public_df, public_df):
    size = ""
    rang = tuple()
    psyh = []
    if indicator_no(text, feature_name, person_public_df, public_df) == 1:
        size = "Нет"
        rang = range_no(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_little(text, feature_name, person_public_df, public_df) == 1:
        size = "Мало"
        rang = range_little(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_less_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Меньше среднего"
        rang = range_less_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Среднее"
        rang = range_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_greater_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Больше среднего"
        rang = range_greater_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_large(text, feature_name, person_public_df, public_df) == 1:
        size = "Много"
        rang = range_large(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
    return size, rang, psyh



def dict_with_feature_range_temp(text, feature_name, person_public_df, public_df):
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    res_dict["Значение"] = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    size, rang, psyh = size_range_temp_of_feature_value(text, feature_name, person_public_df, public_df)
    res_dict["Количество"] = size
    res_dict["Диапазон"] = rang
    res_dict["Темперамент"] = psyh
    return res_dict

def df_with_features_ranges_temp_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(person_public_df._get_numeric_data().columns.values)
    list_of_semantic_role_features = list(person_public_semantic_role_df._get_numeric_data().columns.values)
    #list_of_morph_features = ["Прошедшее, %", "Настоящее, %", "Будущее, %", "Абсолютное, %"]
    for feature_name in list_of_text_features:
        dict_text = dict_with_feature_range_temp(text, feature_name, person_public_df, public_df)
        res_df = res_df.append(dict_text, ignore_index=True)
    for feature_name in list_of_semantic_role_features:
        dict_semantic = dict_with_feature_range_temp(text, feature_name, person_public_semantic_role_df, public_semantic_role_df)
        res_df = res_df.append(dict_semantic, ignore_index=True)
    # for feature_name in list_of_morph_features:
    #     dict_morph = dict_with_feature_range_temp(text, feature_name, person_morph_df, morph_df)
    #     res_df = res_df.append(dict_morph, ignore_index=True)
    cols = ["Название характеристики", "Значение", "Количество", "Диапазон", "Темперамент"]
    res_df = res_df[cols]
    return res_df

def temp_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df):
    print(text)
    dict1 = dict_with_temp_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df)
    dict_res = {k[0]: [k[1]] for k in dict1.items()}
    dict1_norm = dict_norm(dict1)
    dict_res_norm = {k[0]: [k[1]] for k in dict1_norm.items()}
    display(Markdown("## * __Значение = Количество активировавшихся индикаторов темперамента__"))
    display(pd.DataFrame.from_dict(dict_res))
    display(Markdown("## * __Значение = Нормированный по l1-норме предыдущий вектор__"))
    display(pd.DataFrame.from_dict(dict_res_norm))
    print("\n")

def dict_with_temp_by_text(text, person_public_df, person_public_semantic_role_df, public_df, public_semantic_role_df):
    df = df_with_features_ranges_temp_by_text(text, person_public_df, person_public_semantic_role_df, pd.DataFrame(), public_df, public_semantic_role_df, pd.DataFrame())
    dict_res = Counter(np.concatenate(df['Темперамент']))
    return dict_res

def dict_with_sample_ranges_and_temp(feature_name, public_df, public_semantic_role_df):
    list_of_all_types = ["Меланхолик", "Холерик", "Флегматик", "Сангвиник"]
    res_dict = {}
    res_dict["Название характеристики"] = feature_name
    for psyh_name in list_of_all_types:
        res_dict[psyh_name] = range_with_size_by_temp_and_feat_name(feature_name, psyh_name, public_df, public_semantic_role_df)
    return res_dict

def df_with_sample_ranges_and_temp(public_df, public_semantic_role_df):
    res_df = pd.DataFrame()
    list_morphology_features = list(public_df._get_numeric_data().columns.values)
    list_semantic_role_features = list(public_semantic_role_df._get_numeric_data().columns.values)
    list_of_text_features = list_morphology_features + list_semantic_role_features
    for feature_name in list_of_text_features:
        dict_feat = dict_with_sample_ranges_and_temp(feature_name, public_df, public_semantic_role_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    cols = ["Название характеристики", "Меланхолик", "Холерик", "Флегматик", "Сангвиник"]
    res_df = res_df[cols]
    return res_df

def range_with_size_by_temp_and_feat_name(feature_name, psyh_name, public_df, public_semantic_role_df):
    if feature_name in list(public_df.columns.values):
        for size, func in zip(["Мало", "Меньше среднего", "Среднее", "Больше среднего", "Много"],
                       [range_little, range_less_mean, range_mean, range_greater_mean, range_large]):
            if psyh_name in temp_by_feature_name_and_size(feature_name, size):
                return f"{size}: {func(feature_name, public_df)}"
    elif feature_name in list(public_semantic_role_df.columns.values):
        for size, func in zip(["Мало", "Меньше среднего", "Среднее", "Больше среднего", "Много"],
                   [range_little, range_less_mean, range_mean, range_greater_mean, range_large]):
            if psyh_name in temp_by_feature_name_and_size(feature_name, size):
                return f"{size}: {func(feature_name, public_semantic_role_df)}"         
    return "-"

def df_with_features_ranges_and_temp_by_text(text, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df):
    res_df = pd.DataFrame()
    list_of_text_features = list(person_public_df._get_numeric_data().columns.values)
    list_of_semantic_role_features = list(person_public_semantic_role_df._get_numeric_data().columns.values)
    list_of_morph_features = list(person_morph_df._get_numeric_data().columns.values)
    for feature_name in list_of_text_features:
        dict_feat = dict_with_feature_values_range_temp_by_text(text, feature_name, person_public_df, public_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    for feature_name in list_of_semantic_role_features:
        dict_feat = dict_with_feature_values_range_temp_by_text(text, feature_name, person_public_semantic_role_df, public_semantic_role_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    for feature_name in list_of_morph_features:
        dict_feat = dict_with_feature_values_range_temp_by_text(text, feature_name, person_morph_df, morph_df)
        res_df = res_df.append(dict_feat, ignore_index=True)
    cols = ["Название характеристики", "Меланхолик", "Холерик", "Флегматик", "Сангвиник"]
    res_df = res_df[cols]
    return res_df

def dict_with_feature_values_range_temp_by_text(text, feature_name, person_public_df, public_df):
    res_dict = {}
    size, rang, psyh = size_range_temp_of_feature_value(text, feature_name, person_public_df, public_df)
    feature_value = person_public_df[person_public_df["Текст"] == text][feature_name].values[0]
    list_of_all_types = ["Меланхолик", "Холерик", "Флегматик", "Сангвиник"]
    res_dict["Название характеристики"] = feature_name
    for psyh_type in psyh:
        res_dict[psyh_type] = f"{size}: {feature_value} из {rang}"
    for psyh_type in list_of_all_types:
        if psyh_type not in psyh:
            res_dict[psyh_type] = "-"
    return res_dict

def size_range_temp_of_feature_value(text, feature_name, person_public_df, public_df):
    size = ""
    rang = tuple()
    psyh = []
    if indicator_no(text, feature_name, person_public_df, public_df) == 1:
        size = "Нет"
        rang = range_no(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_little(text, feature_name, person_public_df, public_df) == 1:
        size = "Мало"
        rang = range_little(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_less_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Меньше среднего"
        rang = range_less_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Среднее"
        rang = range_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_greater_mean(text, feature_name, person_public_df, public_df) == 1:
        size = "Больше среднего"
        rang = range_greater_mean(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
        
    elif indicator_large(text, feature_name, person_public_df, public_df) == 1:
        size = "Много"
        rang = range_large(feature_name, public_df)
        psyh = temp_by_feature_name_and_size(feature_name, size)
    return size, rang, psyh