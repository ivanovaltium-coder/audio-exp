import pyaudio
import wave
import numpy as np
import time
import os
from typing import Optional, Tuple, List, Dict, Any
import config


def get_asio_device_id(target_name_part="Steinberg"):
    """
    Ищет устройство ASIO Steinberg и возвращает его ID.
    Если не находит, возвращает None.
    """
    p = pyaudio.PyAudio()
    asio_api_idx = -1

    # 1. Ищем индекс Host API ASIO
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_idx = i
            break

    if asio_api_idx == -1:
        p.terminate()
        return None

    # 2. Ищем устройство внутри ASIO
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        # Проверяем, принадлежит ли устройство к ASIO API
        if dev_info['hostApi'] == asio_api_idx:
            # Проверяем имя и наличие входных каналов
            if (target_name_part.lower() in dev_info['name'].lower()) and (dev_info['maxInputChannels'] > 0):
                print(f"✅ Найдено ASIO устройство: {dev_info['name']} (ID: {i})")
                p.terminate()
                return i

    p.terminate()
    return None


def list_asio_devices():
    """Выводит список всех устройств, доступных через ASIO."""
    p = get_pyaudio_instance()
    asio_api_idx = -1

    print("\n=== ПОИСК УСТРОЙСТВА ЧЕРЕЗ ASIO API ===")

    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_idx = i
            break

    if asio_api_idx == -1:
        print("❌ Ошибка поиска: ❌ Драйвер ASIO не найден в системе PyAudio!")
        print("   Убедитесь, что установлен Yamaha Steinberg USB ASIO.")
        p.terminate()
        return

    print(f"✅ ASIO найден! (Host API ID: {asio_api_idx})\n")

    found_count = 0
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['hostApi'] == asio_api_idx:
            print(f"ID {i}: {dev_info['name']}")
            print(f"   Входов: {dev_info['maxInputChannels']}")
            if dev_info['maxInputChannels'] >= 4:
                print("   🎯 ЭТО УСТРОЙСТВО ПОДХОДИТ ДЛЯ МНОГОКАНАЛЬНОЙ ЗАПИСИ")
            print("-" * 30)
            found_count += 1

    if found_count == 0:
        print("   Нет устройств в ASIO. Проверьте панель управления звуком.")

    p.terminate()


def record_audio(
        duration: float = None,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None,
        format_type: int = None
) -> np.ndarray:
    """
    Записывает аудио через PyAudio с прямым указанием параметров ASIO.
    """
    # Параметры по умолчанию из config или жестко заданные для UR44C
    if duration is None: duration = getattr(config, 'RECORD_DURATION', 5.0)
    if channels is None: channels = getattr(config, 'NUM_CHANNELS', 8)
    if sample_rate is None: sample_rate = getattr(config, 'SAMPLE_RATE', 96000)
    if format_type is None: format_type = getattr(config, 'AUDIO_FORMAT', pyaudio.paInt24)

    p = get_pyaudio_instance()

    # Определение устройства
    target_device_id = device_index
    if target_device_id is None:
        target_device_id = get_asio_device_id("Steinberg")
        if target_device_id is None:
            # Если ASIO не найден, пробуем найти любое устройство с нужным кол-вом каналов
            print("⚠️ Попытка записи без явного ASIO...")
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev['maxInputChannels'] >= channels and "Steinberg" in dev['name']:
                    target_device_id = i
                    break

    if target_device_id is None:
        raise ValueError("Не удалось найти подходящее аудиоустройство. Запустите --list-asio.")

    dev_info = p.get_device_info_by_index(target_device_id)
    print(f"🎙️ Запись: Устройство '{dev_info['name']}' (ID: {target_device_id})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Формат: {format_type}, Длительность: {duration} сек")

    frames = []

    try:
        stream = p.open(
            format=format_type,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=target_device_id,
            frames_per_buffer=getattr(config, 'BUFFER_SIZE', 4096)
        )

        print("   🟢 Запись началась...")
        chunks_to_record = int(sample_rate / getattr(config, 'BUFFER_SIZE', 4096) * duration)

        for _ in range(chunks_to_record):
            data = stream.read(getattr(config, 'BUFFER_SIZE', 4096), exception_on_overflow=False)
            frames.append(data)

        print("   🔴 Запись завершена.")
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        raise

    p.terminate()

    # Конвертация в numpy
    raw_data = b''.join(frames)

    # Обработка формата (paInt24 требует особой обработки)
    if format_type == pyaudio.paInt24:
        # Преобразование 3 байт в 4 байта (int32)
        # Разбиваем байты на тройки
        samples = []
        for i in range(0, len(raw_data), 3 * channels):
            chunk = raw_data[i:i + 3 * channels]
            if len(chunk) < 3 * channels: break

            for ch in range(channels):
                idx = ch * 3
                # Байты в little-endian: [B0, B1, B2] -> знаковое 24 бит
                b0, b1, b2 = chunk[idx], chunk[idx + 1], chunk[idx + 2]
                val = b0 | (b1 << 8) | (b2 << 16)
                if val & 0x800000:  # Отрицательное число
                    val -= 0x1000000
                samples.append(val)

        audio_array = np.array(samples, dtype=np.int32)
        # Reshape
        total_samples = len(audio_array) // channels
        audio_array = audio_array[:total_samples * channels].reshape(-1, channels)
        # Нормализация (максимум для 24 бит = 8388607)
        return audio_array.astype(np.float32) / 8388607.0

    elif format_type == pyaudio.paInt16:
        audio_array = np.frombuffer(raw_data, dtype=np.int16)
        audio_array = audio_array.reshape(-1, channels)
        return audio_array.astype(np.float32) / 32768.0

    else:
        # Для int32
        audio_array = np.frombuffer(raw_data, dtype=np.int32)
        audio_array = audio_array.reshape(-1, channels)
        return audio_array.astype(np.float32) / 2147483647.0


def check_microphone(device_index: Optional[int] = None, duration: float = 2.0):
    """Проверка микрофона с записью и анализом уровня."""
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio + ASIO)")
    print("=" * 60)

    try:
        # Пробуем записать 8 каналов
        audio_data = record_audio(
            duration=duration,
            device_index=device_index,
            channels=8,  # Жестко 8 каналов для проверки
            sample_rate=96000,
            format_type=pyaudio.paInt24
        )

        # Анализируем первые 4 канала (микрофоны)
        print("\n--- Анализ уровней (каналы 1-4) ---")
        for ch in range(min(4, audio_data.shape[1])):
            channel_data = audio_data[:, ch]
            rms = np.sqrt(np.mean(channel_data ** 2))
            db = 20 * np.log10(rms + 1e-10)
            status = "✅ OK" if db > -40 else "❌ Тишина"
            print(f"Канал {ch + 1}: {db:.1f} дБ {status}")

        return {"success": True, "data": audio_data}

    except Exception as e:
        print(f"\n❌ Ошибка проверки: {e}")
        return {"success": False, "error": str(e)}
