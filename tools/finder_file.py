import os


def find_latest_file(folder_path):
    """
    Функция находит актуальный вес для инференса модели
    Принимает на вход имя директории, в которой хранятся веса
    Возвращает имя акутального веса, с которым будет инициализирована модель для инференса
    """

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    newest_file = max(files, key=os.path.getmtime)
    return newest_file
