# main.py
import argparse
from recognition.recognizer import DroneRecognizer


def print_result(class_name, confidence):
    """Простой колбэк для вывода результатов в консоль."""
    print(f"{class_name}: {confidence:.3f}")


def main():
    parser = argparse.ArgumentParser(description='DronePrint — акустическая детекция дронов')
    parser.add_argument('--file', type=str, help='Распознать файл вместо микрофона')
    args = parser.parse_args()

    recognizer = DroneRecognizer()

    if args.file:
        class_name, conf = recognizer.recognize_file(args.file)
        print(f"Файл: {args.file}")
        print(f"Результат: {class_name} (уверенность: {conf:.3f})")
    else:
        recognizer.recognize_stream(print_result)


if __name__ == '__main__':
    main()
