# main.py
import argparse
from recognition.recognizer import DroneRecognizer


def print_result(class_name, confidence, probs):
    print(f"pred: {class_name}, conf: {confidence:.3f}  [bg: {probs[0]:.3f}, drone: {probs[1]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description='DronePrint — акустическая детекция дронов')
    parser.add_argument('--file', type=str, help='Распознать аудиофайл (WAV) вместо микрофона')
    parser.add_argument('--list-devices', action='store_true', help='Показать список аудиоустройств и выйти')
    parser.add_argument('--device', type=int, default=None, help='Индекс устройства ввода для микрофона')
    args = parser.parse_args()

    # Если нужно просто показать устройства
    if args.list_devices:
        # Создаём временный объект recognizer, чтобы воспользоваться его методом
        # Но recognizer требует загруженную модель, а нам она не нужна для списка устройств.
        # Можно вызвать метод класса напрямую, импортировав функцию из recognizer?
        # Проще импортировать sounddevice здесь.
        import sounddevice as sd
        print(sd.query_devices())
        print("\nУстройство ввода по умолчанию:", sd.default.device[0])
        return

    # Инициализируем распознаватель (загружает модель)
    recognizer = DroneRecognizer()

    if args.file:
        class_name, conf = recognizer.recognize_file(args.file)
        print(f"Файл: {args.file}")
        print(f"Результат: {class_name} (уверенность: {conf:.3f})")
    else:
        # Запуск непрерывного распознавания с микрофона
        recognizer.recognize_stream(print_result, device=args.device)


if __name__ == '__main__':
    main()
