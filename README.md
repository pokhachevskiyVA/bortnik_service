# Библиотека для анализа текстов

## Установка (требуется Python 3.8 и новее, 8GB RAM)   

```
git clone https://github.com/pokhachevskiyVA/bortnik_service.git

cd bortnik_service

```

    # Далее следует скопировать из папки Soft-Skill-Dev/21_nov
    # https://drive.google.com/drive/folders/1T1NuaU1qPQsyAM_i55AsJPrsGA28EZ5j?usp=sharing
    # на Google Drive папку stanza_resources/
    # в папку клонированного репозитория:
    # ./analytics_lib/notebooks/

    # Также следует скачать ресурсы для elmo модели
    # по ссылке http://vectors.nlpl.eu/repository/20/212.zip,
    # разархивировать их и поместить в ./analytics_lib/notebooks/elmo_resources

```
docker build -t service .

docker run -it --rm -p 8080:8080 service 

```

    # Далее, переходим на http://localhost:8080/docs# и пользуемся сервисом



## Документация

Пользовательскую документацию можно получить по [ссылке](./analytics_lib/docs/index.md).


## Структура репозитория

    ├── analytics_lib       <- основная папка с проектом
        |
        ├── data                <- кэш-файлы, необходимые для запуска ноутбуков
        |
        ├── docs                <- документация к проекту
        │
        |
        ├── nlp_texts           <- скрипты для обработки текстов
        │
        ├── notebooks           <- ноутбуки с экспериментами по текстам


## Использование

С признаками, извлекаемыми из конкретного текста, можно ознакомиться в ноутбуке [psychotyping_by_text.ipynb](./analytics_lib/notebooks/psychotyping_by_text.ipynb)


## Возможные ошибки

С путями решения возможных ошибок при установке и использовании библиотеки можно ознакомиться в [FAQ](./analytics_lib/docs/FAQ.md)