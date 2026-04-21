import argparse
import sys
import os
import config
from audio.recorder import record_audio, check_microphone, get_asio_device_id


def main():
    parser = argparse.ArgumentParser(description="Акустическая система обнаружения БПЛА (PyAudio Edition)")

    parser.add_argument("--list-asio", action="store_true", help="Список всех устройств (поиск 8 каналов)")
    parser.add_argument("--check-mic", action="store_true", help="Проверка микрофона")
    parser.add_argument("--device", type=int, help="Индекс устройства (ручной выбор)")
    parser.add_argument("--duration", type=float, default=None, help="Длительность записи")
    parser.add_argument("--detect", action="store_true", default=True, help="Режим детекции")

    args = parser.parse_args()

    if args.list_asio:
        print("\n=== ПОИСК УСТРОЙСТВА ЧЕРЕЗ ASIO API ===")
        try:
            dev_id = get_asio_device_id()
            print(f"✅ Успешно найдено устройство ASIO с ID: {dev_id}")
            print("Используйте этот ID для запуска записи: python main.py --device <ID>")
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
        return

    if args.check_mic:
        res = check_microphone(device_index=args.device)
        sys.exit(0 if res.get('success') else 1)

    if args.detect:
        print("🚀 Запуск системы...")
        # Здесь будет логика загрузки модели и записи
        # Для теста просто запишем звук
        try:
            audio = record_audio(duration=args.duration or 3.0, device_index=args.device)
            print(f"✅ Записано: {audio.shape}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
