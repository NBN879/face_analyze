from deepface import DeepFace
import json


# Face verification function
def face_verify(img_1, img_2):
    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2)

        with open('result.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        # return result_dict

        if result_dict.get('verified'):
            return "Проверка пройдена. Пропустить."
        return "Нарушитель! Задержать!"

    except Exception as _ex:
        return _ex


# Face recognition function
def face_recogn(img_path, db_path):
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path)
        result = result.values.tolist()

        return result
    except Exception as _ex:
        return _ex


# Human data recognition function (gender, age, ...)
def face_analyze(img_path):
    try:
        result_dict = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])

        with open('face_analyze.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        print(f'[+] Age: {result_dict.get("age")}')
        print(f'[+] Gender: {result_dict.get("gender")}')

        print('[+] Race:')
        for k, v in result_dict.get('race').items():
            print(f'{k} - {round(v, 3)}%')

        print('[+] Emotions:')
        for k, v in result_dict.get('emotion').items():
            print(f'{k} - {round(v, 3)}%')

        return result_dict

    except Exception as _ex:
        return _ex


def main():
    # print(face_verify(img_1='faces/dr_5.jpg', img_2='faces/dr_2.jpg'))
    # print(face_recogn(img_path='faces/50-cent.jpg', db_path='Julia'))
    print(face_analyze(img_path='faces/jim.jpg'))


if __name__ == "__main__":
    main()