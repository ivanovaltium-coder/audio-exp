import argparse
import sys
import os
import config
from audio.recorder import record_audio, check_microphone, list_all_devices_detailed, find_best_device


def main():
    parser = argparse.ArgumentParser(description="Акустическая система обнаружения БПЛА (PyAudio Edition)")

    parser.add_argument("--list-asio", action="store_true", help="Список всех устройств (поиск 8 каналов)")
    parser.add_argument("--check-mic", action="store_true", help="Проверка микрофона")
    parser.add_argument("--device", type=int, help="Индекс устройства (ручной выбор)")
    parser.add_argument("--duration", type=float, default=None, help="Длительность записи")
    parser.add_argument("--detect", action="store_true", default=True, help="Режим детекции")

    args = parser.parse_args()

    if args.list_asio:
        devices = list_all_devices_detailed()
        print(f"\n{'ID':<5} | {'Название':<40} | {'Входы':<5} | {'Частота':<8} | {'Host API'}")
        print("-" * 90)
        for d in devices:
            name = (d['name'][:37] + '...') if len(d['name']) > 40 else d['name']
            print(f"{d['index']:<5} | {name:<40} | {d['channels']:<5} | {d['sample_rate']:<8} | {d['host_api']}")

        print("\n💡 Совет: Ищите устройство с 8 входами. Если Steinberg показывает только 2, ")
        print("   попробуйте устройства с названием 'Input' или проверьте настройки dspMixFX.")
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
