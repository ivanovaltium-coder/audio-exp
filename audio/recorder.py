import pyaudio
import wave
import numpy as np
import time
import os
from typing import Optional, Dict, Any, List
import config


def get_pyaudio_instance() -> pyaudio.PyAudio:
    """Создает и возвращает экземпляр PyAudio."""
    return pyaudio.PyAudio()


def get_asio_device_id(device_name: str = "Steinberg") -> Optional[int]:
    """
    Ищет устройство ASIO (Yamaha Steinberg) и возвращает его ID.
    """
    p = get_pyaudio_instance()
    asio_api_idx = -1

    # 1. Находим индекс Host API ASIO
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_idx = i
            break

    if asio_api_idx == -1:
        print("❌ Драйвер ASIO не найден в системе PyAudio!")
        p.terminate()
        return None

    # 2. Ищем устройство внутри ASIO API
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        # Проверяем, принадлежит ли устройство к ASIO API и имеет ли входы
        if dev_info['hostApi'] == asio_api_idx and dev_info['maxInputChannels'] > 0:
            if device_name.lower() in dev_info['name'].lower():
                print(f"✅ Найдено ASIO устройство: {dev_info['name']} (ID: {i})")
                print(f"   Доступно входов: {dev_info['maxInputChannels']}")
                p.terminate()
                return i

    print(f"⚠️ Устройство '{device_name}' не найдено в ASIO.")
    # Если точное имя не найдено, вернем первое попавшееся ASIO устройство с входами
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['hostApi'] == asio_api_idx and dev_info['maxInputChannels'] >= 4:
            print(f"💡 Используем альтернативное устройство: {dev_info['name']} (ID: {i})")
            p.terminate()
            return i

    p.terminate()
    return None


def list_asio_devices():
    """Выводит список всех устройств ASIO."""
    print("\n=== ПОИСК УСТРОЙСТВА ЧЕРЕЗ ASIO API ===")
    p = get_pyaudio_instance()
    asio_api_idx = -1

    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_idx = i
            break

    if asio_api_idx == -1:
        print("❌ Ошибка поиска: ❌ Драйвер ASIO не найден в системе PyAudio!")
        print("Убедитесь, что установлен Yamaha Steinberg USB ASIO и закрыт в других программах.")
        p.terminate()
        return

    found = False
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['hostApi'] == asio_api_idx:
            found = True
            print(f"ID: {i} | {dev_info['name']}")
            print(f"   Входов: {dev_info['maxInputChannels']}, Частота: {dev_info['defaultSampleRate']}")

    if not found:
        print("❌ Устройства ASIO не найдены.")

    p.terminate()


def record_audio(
        duration: float = None,
        device_index: int = None,
        channels: int = None,
        sample_rate: int = None
) -> np.ndarray:
    """
    Записывает аудио через ASIO драйвер.
    По умолчанию: 8 каналов, 96000 Гц, 24 бита.
    """
    # Настройки по умолчанию из config или жестко заданные для ASIO
    if duration is None:
        duration = getattr(config, 'RECORD_DURATION', 5.0)
    if channels is None:
        channels = 8  # Критично для UR44C
    if sample_rate is None:
        sample_rate = 96000  # Критично для UR44C

    p = get_pyaudio_instance()

    # Определение устройства
    if device_index is None:
        device_index = get_asio_device_id("Steinberg")
        if device_index is None:
            raise ValueError("Не удалось найти устройство ASIO автоматически. Укажите --device <ID>")

    # Проверка существования устройства
    try:
        dev_info = p.get_device_info_by_index(device_index)
        print(f"🎙️ Запись: Устройство '{dev_info['name']}' (ID: {device_index})")
        print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
        print(f"   Формат: 24 бита (paInt24)")
    except IOError:
        raise ValueError(f"Устройство с ID {device_index} не найдено.")

    frames = []
    audio_format = pyaudio.paInt24
    chunk = 4096  # Размер буфера как в рабочем примере

    stream = None
    try:
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk
        )

        print("   🟢 Запись началась...")
        chunks_to_record = int(sample_rate / chunk * duration)

        for _ in range(chunks_to_record):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        print("   🔴 Запись завершена.")

    except Exception as e:
        print(f"❌ Ошибка во время записи: {e}")
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
        raise
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    # Конвертация байтов в numpy массив (для 24 бит это сложнее)
    # PyAudio возвращает 24 бита в 3 байтах.
    # Для простоты классификации часто приводят к 32 битам или обрабатывают байты напрямую.
    # Здесь сделаем конвертацию в float32 [-1, 1] для совместимости с sklearn/librosa.

    raw_data = b''.join(frames)

    # Обработка 24-битных данных (3 байта на сэмпл)
    # Преобразуем в numpy array uint8, затем переупакуем в int32
    samples_count = len(raw_data) // (channels * 3)
    audio_array = np.zeros((samples_count * channels), dtype=np.int32)

    # Быстрая векторизованная конвертация (little-endian)
    # Байты: [B0, B1, B2] -> Int32: [0, B2, B1, B0] (с знаковым расширением)
    raw_bytes = np.frombuffer(raw_data, dtype=np.uint8)
    raw_bytes = raw_bytes.reshape(-1, 3)

    # Сдвигаем байты и собираем 32-битное число
    # B0 - младший, B2 - старший (знаковый)
    audio_array = (raw_bytes[:, 0].astype(np.int32) |
                   (raw_bytes[:, 1].astype(np.int32) << 8) |
                   (raw_bytes[:, 2].astype(np.int32) << 16))

    # Знаковое расширение для 24 бит (если старший бит 23-го разряда равен 1)
    mask = 1 << 23
    audio_array = (audio_array ^ mask) - mask

    # Reshape в (samples, channels)
    audio_array = audio_array.reshape(-1, channels).astype(np.float32)

    # Нормализация (максимальное значение 24-битного числа = 2^23 - 1)
    audio_array /= 8388607.0

    return audio_array


def check_microphone(
        device_index: Optional[int] = None,
        duration: float = 2.0
) -> Dict[str, Any]:
    """Проверка микрофона."""
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio + ASIO)")
    print("=" * 60)

    if device_index is None:
        device_index = get_asio_device_id("Steinberg")
        if device_index is None:
            return {"success": False, "error": "ASIO device not found"}

    try:
        # Записываем 8 каналов, но анализируем первый
        audio_data = record_audio(
            duration=duration,
            device_index=device_index,
            channels=8,
            sample_rate=96000
        )

        # Анализ первого канала
        channel_0 = audio_data[:, 0]
        rms = np.sqrt(np.mean(channel_0 ** 2))
        db_level = 20 * np.log10(rms + 1e-10)
        peak = np.max(np.abs(channel_0))
        peak_db = 20 * np.log10(peak + 1e-10)

        threshold_db = getattr(config, 'MIC_CHECK_THRESHOLD_DB', -40)

        print("\n" + "-" * 60)
        print(f"РЕЗУЛЬТАТЫ (Канал 1 из 8):")
        print("-" * 60)
        print(f"Средний уровень (RMS): {db_level:.2f} дБ")
        print(f"Пиковый уровень: {peak_db:.2f} дБ")

        success = db_level > threshold_db

        if success:
            print("\n✅ МИКРОФОН РАБОТАЕТ! (Сигнал > порога)")
        else:
            print("\n❌ СИГНАЛ СЛИШКОМ ТИХИЙ. Проверьте Gain и подключение.")

        return {"success": success, "rms_db": db_level}

    except Exception as e:
        print(f"\n❌ Ошибка проверки: {e}")
        return {"success": False, "error": str(e)}
