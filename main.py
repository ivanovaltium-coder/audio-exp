# main.py
import argparse
from recognition.recognizer import DroneRecognizer
from audio.recorder import check_microphone, list_devices


def print_result(class_name, confidence, probs):
    print(f"pred: {class_name}, conf: {confidence:.3f}  [bg: {probs[0]:.3f}, drone: {probs[1]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description='DronePrint — акустическая детекция дронов')
    parser.add_argument('--file', type=str, help='Распознать аудиофайл (WAV) вместо микрофона')
    parser.add_argument('--list-devices', action='store_true', help='Показать список аудиоустройств и выйти')
    parser.add_argument('--device', type=int, default=None, help='Индекс устройства ввода для микрофона')
    parser.add_argument('--check-mic', action='store_true', help='Проверить работу микрофона и выйти')
    parser.add_argument('--mic-duration', type=float, default=2.0, help='Длительность проверки микрофона в секундах')
    args = parser.parse_args()

    # Если нужно просто показать устройства
    if args.list_devices:
        list_devices()
        return

    # Если нужно проверить микрофон
    if args.check_mic:
        result = check_microphone(
            device=args.device,
            duration=args.mic_duration,
            verbose=True
        )
        if result['working']:
            print("\n✓ Проверка пройдена успешно!")
        else:
            print("\n✗ Проверка не пройдена. Требуется настройка оборудования.")
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
