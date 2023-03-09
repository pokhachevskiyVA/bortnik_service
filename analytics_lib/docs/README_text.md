## Описание методов класса TextProcessor из [text.py](../nlp_texts/text.py), необходимых для результатов по психотипированию

**compute_one_df(userId, df_name)**

    Возвращает таблицу person_{df_name} с посчитанными фичам по текстам token'а {userId}.
    Для полной обработки нового token'а в качестве df_name необходимо подавать поочерёдно: 
    public_df, public_semantic_role_df, morph_df
    и сохранять полученные результаты в таблицы person_public_df, person_public_semantic_role_df, person_morph_df. 
    Последние таблицы необходимы для дальнейшего отчёта по token'у.

**compute_one_df_glue(userId, df_name)**

    Возвращает таблицу person_{df_name} с посчитанными фичам по склеенным текстам token'а {userId}.
    Для обработки склеенного текста нового token'а в качестве df_name необходимо подавать поочерёдно: 
    public_df, public_semantic_role_df, morph_df.
    Последние таблицы необходимы для дальнейшего отчёта по token'у.
    
**tables_per_text_by_token(userId, person_public_df, public_df)**

    Возвращает список таблиц с фичами и средними, каждая из которых отвечает отдельному тексту человека с token'ом userId. 
    На вход так же требуется таблица, с посчитанными фичами по всем текстам public_df.
    Внутри этой функции есть список list_of_needed_rows с номерами признаков,
    которые интересуют нас в текущей версии психотипирования.
    Для добавления новых признаков, необходимо добавить их номера в данный список.

**tables_per_text_by_token_psyh(userId, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df)**

    Возвращает список таблиц со значениями характеристик, диапазонов и психотипов, каждая из которых отвечает отдельному
    тексту человека с token'ом userId. 
    На вход так же требуются таблицы, с посчитанными фичами по всем текстам всех людей:
    public_df, public_semantic_role_df, morph_df,
    а также таблицы с посчитанными фичами по всем текстам данного token'а: 
    person_public_df, person_public_semantic_role_df, person_morph_df.
    
    Чтобы добавить новые характеристики в таблицу, необходимо изменить функцию
    df_with_features_ranges_psyh_by_text из psychotype.py,
    к которой обращается данная (добавить название соответствующей характеристики в список).
    А также необходимо дописать распределение новой характеристики по психотипам в функции
    psyh_by_feature_name_and_size(feature_name, size) из psychotype.py по аналогии.

**tables_per_text_by_token_psyh_cosy(userId, person_public_df, person_public_semantic_role_df, person_morph_df, public_df, public_semantic_role_df, morph_df)**

    Возвращает список таблиц со значениями характеристик, диапазонов и психотипов, каждая из которых отвечает отдельному
    тексту человека с token'ом userId (в удобном для аналитика виде). 
    На вход так же требуются таблицы, с посчитанными фичами по всем текстам всех людей:
    public_df, public_semantic_role_df, morph_df,
    а также таблицы с посчитанными фичами по всем текстам данного token'а:
    person_public_df, person_public_semantic_role_df, person_morph_df.
    Чтобы добавить новые характеристики в таблицу, необходимо изменить функцию
    df_with_features_ranges_psyh_by_text из psychotype.py,
    к которой обращается данная (добавить название соответствующей характеристики в список).
    А также необходимо дописать распределение новой характеристики по психотипам в функции
    psyh_by_feature_name_and_size(feature_name, size) из psychotype.py по аналогии.

**tables_per_text_by_token_semantic_role(userId, person_public_semantic_role_df, public_semantic_role_df)**

    Возвращает список таблиц с анализом агенсных конструкций и предикатов, каждая из которых отвечает отдельному
    тексту человека с token'ом userId (в удобном для аналитика виде).
    На вход требует таблицу public_semantic_role_df с посчитанными semantic_role фичами по всей выборке,
    а также person_public_semantic_role_df - таблицу с посчитанными semantic_role фичами по текстам данного token'а.

**tables_per_text_by_token_morph(userId, person_morph_df, morph_df)**

    Возвращает список таблиц с анализом времен глаголов, каждая из которых отвечает отдельному
    тексту человека с token'ом userId (в удобном для аналитика виде).
    На вход требует таблицу morph_df с посчитанными временами глаголов по всей выборке,
    а также person_morph_df - таблицу с посчитанными временами глаголов по текстам данного token'а.

**tables_per_text_by_token_figure(userId, person_public_figure_df, public_figure_df)**

    Возвращает список таблиц с анализом фигур в тексте, каждая из которых отвечает отдельному
    тексту человека с token'ом userId (в удобном для аналитика виде).
    На вход требует таблицу public_figure_df с посчитанными фигурами по всей выборке,
    а также person_public_figure_df - таблицу с посчитанными фигурами по текстам данного token'а.

**tables_per_text_by_token_modality(userId, person_public_modality_df, public_modality_df)**

    Возвращает список таблиц с анализом модальностей в тексте, каждая из которых отвечает отдельному
    тексту человека с token'ом userId (в удобном для аналитика виде).
    На вход требует таблицу public_modality_df с посчитанными модальностями по всей выборке,
    а также person_public_modality_df - таблицу с посчитанными модальностями по текстам данного token'а.

**df_with_new_labels_by_token(userId, person_df_for_new_labeling, public_df_for_new_labeling)**

    Возвращает таблицу с новой разметкой от Руслана человека с token'ом userId.
    На вход требует таблицу public_df_for_new_labeling с посчитанной разметкой по всей выборке,
    а также person_df_for_new_labeling - таблицу с посчитанной разметкой по текстам данного token'а.

**table_random_syst_by_token(userId, dict_with_indicators_events, dict_mouse_feat, df_for_rand_syst_indicators)**

    Возвращает таблицу с альтернативными индикаторами шкалы "Хаотичная/Системная" с обратными квантилями 
    человека с token'ом userId (в удобном для аналитика виде).
    На вход требует таблицу df_for_rand_syst_indicators с посчитанными индикаторами по всей выборке,
    а также dict_with_indicators_events - словарь с посчитанными текстовыми индикаторами данного человека и
    dict_mouse_feat - словарь с фичами мышки, посчитанными с помощью API-сервиса.