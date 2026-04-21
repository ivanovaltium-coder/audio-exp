import argparse
import sys
import os
import numpy as np
import config
from audio.recorder import record_audio, check_microphone, get_asio_device_id
from recognition.recognizer import DroneRecognizer


def print_header(text: str):
    """Красивый заголовок."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Система акустического обнаружения БПЛА")

    # Режимы работы
    parser.add_argument("--list-asio", action="store_true", help="Список доступных ASIO устройств")
    parser.add_argument("--check-mic", action="store_true", help="Проверка уровня сигнала микрофонов")
    parser.add_argument("--detect", action="store_true", help="Режим обнаружения (по умолчанию)", default=True)

    # Настройки
    parser.add_argument("--device", type=int, help="ID устройства ASIO (например, 53)")
    parser.add_argument("--duration", type=float, default=5.0, help="Длительность анализа в секундах")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH, help="Путь к файлу модели")
    parser.add_argument("--scaler", type=str, default=config.SCALER_PATH, help="Путь к файлу скалера")

    args = parser.parse_args()

    # --- РЕЖИМ 1: Список устройств ---
    if args.list_asio:
        print_header("СПИСОК УСТРОЙСТВ ASIO")
        dev_id = get_asio_device_id()
        if dev_id:
            print(f"\n💡 Совет: Используйте ID {dev_id} для запуска: python main.py --device {dev_id}")
        return

    # --- РЕЖИМ 2: Проверка микрофона ---
    if args.check_mic:
        print_header("ПРОВЕРКА ОБОРУДОВАНИЯ")
        result = check_microphone(device_index=args.device, duration=2.0)
        if result.get("success"):
            print("\n✅ Система готова к работе!")
            sys.exit(0)
        else:
            print("\n❌ Ошибка оборудования. Проверьте подключение и настройки Gain.")
            sys.exit(1)

    # --- РЕЖИМ 3: Обнаружение БПЛА (Основной) ---
    if args.detect:
        print_header("ЗАПУСК СИСТЕМЫ ОБНАРУЖЕНИЯ БПЛА")

        # 1. Проверка наличия моделей
        if not os.path.exists(args.model):
            print(
                f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл модели не найден!\n   Путь: {args.model}\n   Решение: Запустите 'python train_model.py'")
            sys.exit(1)
        if not os.path.exists(args.scaler):
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Файл скалера не найден!\n   Путь: {args.scaler}")
            sys.exit(1)

        try:
            # 2. Инициализация распознавателя
            print(f"\n🧠 Загрузка модели из: {os.path.basename(args.model)}")
            recognizer = DroneRecognizer(model_path=args.model, scaler_path=args.scaler)

            # 3. Запись аудио
            print(f"\n🎙️ Запись сигнала ({args.duration} сек)...")
            print("   (Ожидайте завершения...)")

            audio_data = record_audio(
                duration=args.duration,
                device_index=args.device,
                output_filename="last_recording.wav"
            )

            if audio_data is None or audio_data.size == 0:
                print("❌ Ошибка: Не удалось получить аудиоданные.")
                sys.exit(1)

            print(f"   ✅ Получено данных: {audio_data.shape} (семплы × каналы)")

            # 4. Подготовка сигнала (берем 1-й канал для классификации)
            # В будущем можно реализовать слияние каналов или пеленгацию
            if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                mono_signal = audio_data[:, 0]  # Берем первый микрофон
                print("   ℹ️ Используется сигнал с Канала 1 (из 8 доступных)")
            else:
                mono_signal = audio_data.flatten()

            # 5. Классификация
            print("\n⏳ Анализ признаков и классификация...")
            prediction, confidence = recognizer.predict(mono_signal)

            # 6. Вывод результата
            print_header("РЕЗУЛЬТАТ АНАЛИЗА")

            if prediction == 1:
                print("🛸 СТАТУС: ОБНАРУЖЕН БПЛА (DRONE DETECTED)")
                print(f"   Уверенность системы: {confidence:.1%}")
                print("   ⚠️ Рекомендуется визуальное подтверждение цели.")
            else:
                print("🌳 СТАТУС: ФОНОВЫЙ ШУМ (NO DRONE)")
                print(f"   Уверенность системы: {confidence:.1%}")
                print("   Сигнал не содержит характерных признаков БПЛА.")

            print("-" * 60)
            print(f"💾 Аудиофайл сохранен: last_recording.wav")
            print_header("СЕАНС ЗАВЕРШЕН")

        except FileNotFoundError as e:
            print(f"\n❌ Ошибка файлов: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Критическая ошибка при выполнении: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
