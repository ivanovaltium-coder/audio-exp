import pyaudio
import numpy as np
import time
import os
import sys
from datetime import datetime

# Импорт наших модулей
from recognition.feature_extractor import FeatureExtractor
from models.classifier import DroneClassifier
import config


def get_asio_devices():
    """Возвращает список устройств ASIO."""
    p = pyaudio.PyAudio()
    asio_devices = []

    # Поиск API ASIO
    asio_api_idx = -1
    for i in range(p.get_host_api_count()):
        api = p.get_host_api_info_by_index(i)
        if "ASIO" in api['name'].upper():
            asio_api_idx = i
            break

    if asio_api_idx == -1:
        print("❌ ASIO драйвер не найден!")
        p.terminate()
        return []

    # Поиск устройств внутри ASIO
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['hostApi'] == asio_api_idx and dev['maxInputChannels'] > 0:
            # Для Steinberg принудительно считаем, что он умеет 96к, даже если API говорит 44.1к
            rate = int(dev['defaultSampleRate'])
            if "steinberg" in dev['name'].lower() or "ur44" in dev['name'].lower():
                rate = 96000

            asio_devices.append({
                'index': dev['index'],
                'name': dev['name'],
                'channels': dev['maxInputChannels'],
                'rate': rate
            })

    p.terminate()
    return asio_devices


def main():
    print("🔍 Поиск ASIO устройств...")
    devices = get_asio_devices()

    if not devices:
        return

    print("\nНайдено устройств:")
    for i, dev in enumerate(devices):
        marker = ">>> " if "steinberg" in dev['name'].lower() else "    "
        print(f"{marker}[{i}] ID: {dev['index']} | {dev['name']} | Частота: {dev['rate']} Гц")

    # Выбор устройства
    choice = input(f"\nВведите номер списка (0-{len(devices) - 1}) или ID устройства (Enter для Steinberg): ").strip()

    selected_dev = None
    if choice == "":
        # Выбираем первый Steinberg или первый в списке
        for d in devices:
            if "steinberg" in d['name'].lower():
                selected_dev = d
                break
        if not selected_dev and devices:
            selected_dev = devices[0]
    elif choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(devices):
            selected_dev = devices[idx]
        else:
            # Может быть введен ID напрямую
            for d in devices:
                if d['index'] == int(choice):
                    selected_dev = d
                    break

    if not selected_dev:
        print("❌ Устройство не выбрано.")
        return

    print(f"\n🧠 Загрузка модели...")
    if not os.path.exists(config.MODEL_PATH):
        print(f"❌ Модель не найдена: {config.MODEL_PATH}")
        return

    classifier = DroneClassifier.load(config.MODEL_PATH, config.SCALER_PATH)
    extractor = FeatureExtractor(sr=selected_dev['rate'])  # Важно: создаем экстрактор под частоту устройства

    print("✅ Модель готова.")

    # Настройки записи
    CHUNK = 4096
    DURATION = 3.0  # Секунд на один цикл
    FORMAT = pyaudio.paInt24
    CHANNELS = min(8, selected_dev['channels'])
    RATE = selected_dev['rate']

    # Проверка соответствия частоты модели и устройства
    # Модель обучена на 96кГц. Если устройство другое, нужно предупреждение.
    if RATE != 96000:
        print(f"⚠️ ВНИМАНИЕ: Частота устройства {RATE} Гц отличается от обучающей (96000 Гц)!")
        print("   Точность может быть низкой. Рекомендуется использовать Steinberg UR44C.")
        # Для теста попробуем запустить, но экстрактор будет работать с этой частотой.
        # Если модель жестко требует 96к, здесь будет ошибка размерности.
        # В идеале: если RATE != 96000, нужно делать ресемплинг перед экстракцией.
        # Но пока оставим как есть, предполагая, что пользователь выберет Steinberg.

    p = pyaudio.PyAudio()

    print("\n" + "=" * 70)
    print("  📡 ПОТОКОВЫЙ МОНИТОРИНГ (REAL-TIME)")
    print("=" * 70)
    print(f"Устройство: {selected_dev['name']} (ID: {selected_dev['index']})")
    print(f"Параметры: {CHANNELS} кан., {RATE} Гц")
    print("Статус: ОЖИДАНИЕ...")
    print("Для остановки нажмите красный квадрат (Stop) в IDE или Ctrl+C")
    print("=" * 70 + "\n")

    stream = None
    cycle_count = 0
    start_time = time.time()

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=selected_dev['index'],
                        frames_per_buffer=CHUNK)

        stream.start_stream()

        while True:
            # Чтение данных
            frames = []
            chunks_needed = int(RATE / CHUNK * DURATION)

            # Читаем буфер
            for _ in range(chunks_needed):
                if stream.is_stopped():
                    stream.start_stream()
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

            if not frames:
                continue

            # Конвертация в numpy (аналогично recorder.py)
            raw_data = b''.join(frames)
            if FORMAT == pyaudio.paInt24:
                samples = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 3)
                padded = np.zeros((samples.shape[0], 4), dtype=np.uint8)
                padded[:, :3] = samples
                mask = (samples[:, 2] & 0x80) != 0
                padded[mask, 3] = 0xFF
                audio_int32 = padded.view(dtype=np.int32).reshape(-1, CHANNELS)
                audio_float = audio_int32.astype(np.float32) / 8388608.0
            else:
                audio_float = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, CHANNELS).astype(np.float32) / 32768.0

            # Берем 1 канал
            signal = audio_float[:, 0]

            # Анализ
            try:
                features = extractor.extract_features_from_array(signal)

                if features is None:
                    print("⚠️ Ошибка: не удалось извлечь признаки (слишком короткий сигнал?)")
                    continue

                prediction, confidence = classifier.predict(features)

                # ВЫВОД РЕЗУЛЬТАТА
                timestamp = datetime.now().strftime("%H:%M:%S")
                status = "🛸 БПЛА!" if prediction == 1 else "🌳 ШУМ"
                color_code = "\033[91m" if prediction == 1 else "\033[92m"  # Красный или Зеленый
                reset_code = "\033[0m"

                print(f"[{timestamp}] {color_code}{status}{reset_code} (Вероятность: {confidence:.1%})")

                # Дополнительно: если дрон, можно писать лог
                if prediction == 1:
                    with open("dron_events.log", "a") as f:
                        f.write(f"{timestamp}, Confidence: {confidence}\n")

            except Exception as e:
                print(f"❌ Ошибка анализа: {e}")
                # Раскомментируйте для отладки, если ошибка постоянная:
                # import traceback
                # traceback.print_exc()

            cycle_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n⏹️ Остановка пользователем...")
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

        total_time = time.time() - start_time
        print("\n📊 Статистика сессии:")
        print(f"   Отработано циклов: {cycle_count}")
        print(f"   Общее время: {total_time:.1f} сек")
        print("   Система остановлена.")


if __name__ == "__main__":
    main()
