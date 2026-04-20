import argparse
import sys
import os
import numpy as np
import config
from audio.recorder import record_audio, check_microphone, list_asio_devices
from recognition.recognizer import DroneRecognizer


def main():
    parser = argparse.ArgumentParser(description="Акустическая система обнаружения БПЛА")

    # Режимы работы
    parser.add_argument("--list-asio", action="store_true", help="Список доступных ASIO устройств")
    parser.add_argument("--check-mic", action="store_true", help="Проверка работы микрофона")
    parser.add_argument("--detect", action="store_true", help="Запуск распознавания (по умолчанию)", default=True)

    # Настройки
    parser.add_argument("--device", type=int, help="Индекс устройства ASIO (получить через --list-asio)")
    parser.add_argument("--duration", type=float, default=config.RECORD_DURATION, help="Длительность записи в секундах")
    parser.add_argument("--channels", type=int, default=None, help="Количество каналов записи (по умолчанию из config)")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Путь к файлу модели")
    parser.add_argument("--scaler", type=str, default=config.SCALER_PATH, help="Путь к файлу скалера")

    args = parser.parse_args()

    # 1. Список устройств
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

    # 2. Проверка микрофона
    if args.check_mic:
        result = check_microphone(device_index=args.device, duration=2.0)
        if result.get("success"):
            print("\n✅ Проверка пройдена успешно!")
            sys.exit(0)
        else:
            print("\n❌ Проверка не пройдена.")
            sys.exit(1)

    # 3. Распознавание (Основной режим)
    if args.detect:
        print("🚀 Запуск системы обнаружения БПЛА...")

        # Проверка наличия модели перед запуском
        if not os.path.exists(args.model):
            print(f"❌ Файл модели не найден: {args.model}")
            print("   Сначала обучите модель (train_model.py) или скачайте готовую.")
            sys.exit(1)
        if not os.path.exists(args.scaler):
            print(f"❌ Файл скалера не найден: {args.scaler}")
            sys.exit(1)

        try:
            # Инициализация распознавателя
            recognizer = DroneRecognizer(model_path=args.model, scaler_path=args.scaler)

            # Запись аудио
            print(f"\n🎙️ Запись звука ({args.duration} сек)...")
            audio_data = record_audio(
                duration=args.duration,
                device_index=args.device,
                channels=args.channels
            )

            print(f"   Получено данных: {audio_data.shape}")

            # Обработка и классификация
            # Если каналов несколько, берем первый (или можно усреднить, зависит от задачи)
            if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                print(f"   Многоканальная запись. Используем канал 1 для анализа.")
                mono_audio = audio_data[:, 0]
            else:
                mono_audio = audio_data.flatten()

            prediction, confidence = recognizer.predict(mono_audio)

            print("\n" + "=" * 50)
            print("РЕЗУЛЬТАТ АНАЛИЗА:")
            print("=" * 50)
            if prediction == 1:  # Предполагаем, что 1 - это дрон
                print(f"🛸 ОБНАРУЖЕН БПЛА (DRONE)!")
            else:
                print(f"🌳 ФОНОВЫЙ ШУМ (NO DRONE)")
            print(f"Уверенность модели: {confidence:.2%}")
            print("=" * 50)

        except FileNotFoundError as e:
            print(f"❌ Ошибка файлов: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"❌ Ошибка конфигурации: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()