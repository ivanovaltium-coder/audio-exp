# main.py
import argparse
import sys
from recognition.recognizer import DroneRecognizer
from audio.recorder import check_microphone, list_asio_devices


def print_result(class_name, confidence, probs):
    print(f"pred: {class_name}, conf: {confidence:.3f}  [bg: {probs[0]:.3f}, drone: {probs[1]:.3f}]")


def main():
    parser = argparse.ArgumentParser(description='DronePrint — акустическая детекция дронов')
    parser.add_argument('--file', type=str, help='Распознать аудиофайл (WAV) вместо микрофона')
    parser.add_argument('--list-asio', action='store_true', help='Показать только ASIO устройства (рекомендуется для UR44C)')
    parser.add_argument('--device', type=int, default=None, help='Индекс устройства ввода для микрофона')
    parser.add_argument('--check-mic', action='store_true', help='Проверить работу микрофона и выйти')
    parser.add_argument('--mic-duration', type=float, default=2.0, help='Длительность проверки микрофона в секундах')
    args = parser.parse_args()

    # Если нужно показать только ASIO устройства
    if args.list_asio:
        print("\n=== ДОСТУПНЫЕ УСТРОЙСТВА ASIO ===")
        devices = list_asio_devices()
        if not devices:
            print("❌ Устройства ASIO не найдены. Проверьте установку драйвера Steinberg.")
        else:
            for dev in devices:
                print(f"ID: {dev['index']} | Название: {dev['name']}")
                print(f"      Каналы: {dev['channels']} | Частота: {dev['default_sample_rate']} Гц")
                print("-" * 40)
        return

    # Если нужно проверить микрофон
    if args.check_mic:
        result = check_microphone(
            device_index=args.device,
            duration=args.mic_duration
        )
        if result.get('success'):
            print("\n✅ Проверка пройдена успешно!")
            sys.exit(0)
        else:
            print("\n❌ Проверка не пройдена. Требуется настройка оборудования.")
            sys.exit(1)

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
