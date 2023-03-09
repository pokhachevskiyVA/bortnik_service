import pandas as pd

def processing_part1_df(part1):
    list_of_labels = [
        "Прилагательное",
        "Нет связи",
        "Есть связь",
        "Не знакомо",
        "Прилагательное1",
        "Нет связи1",
        "Есть связь1",
        "Не знакомо1",
    ]
    part1 = part1.set_axis(list_of_labels, axis=1)
    part1 = part1.dropna(axis="rows")
    df1 = part1.iloc[:, :4]
    df2 = part1.iloc[:, 4:]
    list_of_labels = [
        "Прилагательное",
        "Нет связи",
        "Есть связь",
        "Не знакомо",
    ]
    df2 = df2.set_axis(list_of_labels, axis=1)
    df2 = df2.reset_index(drop=True)
    df = df1.append(
        df2, ignore_index=True
    )  # Общий DataFrame с прилагательными, не относящимися к ощущениям
    return df


def processing_part2_df(part2):
    list_of_labels = [
        "Прилагательное",
        "Нет связи",
        "Есть связь",
        "Не знакомо",
        "Связь с ощущениями",
        "Зрение",
        "Слух",
        "Обоняние",
        "Вкус",
        "Кожное",
    ]
    part2 = part2.set_axis(list_of_labels, axis=1)
    part2 = part2.dropna(axis="rows")
    return part2


def coverage_by_dict(stem_list_of_all_text_adj, merge_list):
    count_adj = len(stem_list_of_all_text_adj)
    coverage = 0  # количество прилагательных в текстах, покрываемых словарем
    for adj in stem_list_of_all_text_adj:
        if adj in merge_list:
            coverage += 1
    res = round(
        coverage / (count_adj if count_adj != 0 else 0), 2
    )
    return res


def list_to_list(x):
    list_of_int = []
    list_of_int.append(int(x[0]))
    list_of_int.append(int(x[1]))
    return list_of_int

def diff_func(list_of_strings):
    diff = list_of_strings.str[0] - list_of_strings.str[1]
    return diff


def get_all_sence_df(df):
    df1 = df.copy()

    df1["Обоняние"] = df1["Обоняние"].str.split("\\")
    df1["Обоняние"] = df1["Обоняние"].apply(list_to_list)
    df_smell = df1[diff_func(df1["Обоняние"]) > 6]
    df_smell = df_smell.reset_index(drop=True)

    df1["Зрение"] = df1["Зрение"].str.split("\\")
    df1["Зрение"] = df1["Зрение"].apply(list_to_list)
    df_sight = df1[diff_func(df1["Зрение"]) > 6]
    df_sight = df_sight.reset_index(drop=True)

    df1["Слух"] = df1["Слух"].str.split("\\")
    df1["Слух"] = df1["Слух"].apply(list_to_list)
    df_hear = df1[diff_func(df1["Слух"]) > 6]
    df_hear = df_hear.reset_index(drop=True)

    df1["Кожное"] = df1["Кожное"].str.split("\\")
    df1["Кожное"] = df1["Кожное"].apply(list_to_list)
    df_skin = df1[diff_func(df1["Кожное"]) > 6]
    df_skin = df_skin.reset_index(drop=True)

    df1["Вкус"] = df1["Вкус"].str.split("\\")
    df1["Вкус"] = df1["Вкус"].apply(list_to_list)
    df_taste = df1[diff_func(df1["Вкус"]) > 6]
    df_taste = df_taste.reset_index(drop=True)

    return (df_smell, df_sight, df_hear, df_skin, df_taste)


def list_to_lower(list_of_words):
    lower_list = []
    for word in list_of_words:
        lower_list.append(word.lower())
    return lower_list

def verbs_df_to_internal_pred_verbs(verbs_df):
    list_feel_verbs = list_to_lower(verbs_df[verbs_df["Чувства"] == "(+)"]["Глагол"].to_list()) # чувства
    list_mind_verbs = list_to_lower(verbs_df[verbs_df["Мысли"] == "(+)"]["Глагол"].to_list()) # мысли
    list_relation_verbs = list_to_lower(verbs_df[verbs_df["Отношение"] == "(+)"]["Глагол"].to_list()) # отношение
    list_modal_verbs = list_to_lower(verbs_df[verbs_df["Модальные"] == "(+)"]["Глагол"].to_list()) # модальные
    list_skills_verbs = list_to_lower(verbs_df[verbs_df["Умения"] == "(+)"]["Глагол"].to_list()) # умения
    return list_feel_verbs, list_mind_verbs, list_relation_verbs, list_modal_verbs, list_skills_verbs

def verbs_df_to_external_pred_verbs(verbs_df):
    list_action_verbs = list_to_lower(verbs_df[verbs_df["Действие"] == "(+)"]["Глагол"].to_list()) # действие
    list_communication_verbs = list_to_lower(verbs_df[verbs_df["Коммуникация"] == "(+)"]["Глагол"].to_list()) # коммуникация
    list_being_verbs = list_to_lower(verbs_df[verbs_df["Бытие"] == "(+)"]["Глагол"].to_list()) # бытие
    list_move_verbs = list_to_lower(verbs_df[verbs_df["Движение"] == "(+)"]["Глагол"].to_list()) # движение
    return list_action_verbs, list_communication_verbs, list_being_verbs, list_move_verbs

def update_verbs_df():
    verbs_df = pd.read_excel("../data/verbs_v1.xlsx")
    verbs_df = verbs_df.fillna("(-)")
    cols = ["Глагол", "Чувства", "Мысли", "Отношение", "Модальные", "Умения", "Действие", "Коммуникация", "Бытие", "Движение"]
    verbs_df = verbs_df[cols]
    verbs_df.to_pickle("../data/verbs_df.pkl")


