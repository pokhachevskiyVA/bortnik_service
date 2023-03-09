## Возможные проблемы и способы их решения

* Во время установки библиотек при запуске команды ```pip install -r requirements.txt``` может возникнуть следующая проблема:
    ```
    [InvalidRequirement]
    Invalid requirement, parse error at "'text res'"
    ```
    
    Для её решения необходимо переустановить пакет ```setuptools```, путём последовательного выполнения двух команд:
    ```
    conda remove setuptools
    conda install setuptools
    ```
    
* При запуске jupyter-ноутбуков из папки [notebooks](../notebooks/) в зависимости от вашей ОС может возникнуть следующая проблема:
    ```
    ModuleNotFoundError: No module named 'analytics_lib'
    ```
    
    Для её решения необходимо сместить системный путь из папки [notebooks](../notebooks/) в папку [analytics_text_research](../../../analytics_text_research/):
    ```
    import sys
    sys.path.append("../..")
    ```
    
* При загрузке ELMO-модели ```model.load(PATH_TO_ELMO)``` (в ноутбуках, где это необходимо) может возникнуть следующая проблема:
    ```
    ValueError: Variable bilm/char_embed already exists, disallowed. 
    Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
    ```
    
    Для её решения необходимо прописать следующие строчки перед загрузкой модели:
    ```
    import tensorflow.compat.v1 as tf
    tf.reset_default_graph()
    ```