import pyaudio
import numpy as np
import time
import config
from typing import Optional, Dict, Any, List


def get_pyaudio_instance():
    return pyaudio.PyAudio()


def list_all_devices_verbose():
    """Выводит подробную информацию обо всех устройствах и их Host API."""
    p = get_pyaudio_instance()
    print("\n=== ПОЛНЫЙ СПИСОК УСТРОЙСТВ И HOST API ===")

    # Вывод Host API
    for i in range(p.get_host_api_count()):
        api = p.get_host_api_info_by_index(i)
        print(f"Host API {i}: {api['name']}")

    print("-" * 40)

    # Вывод устройств
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        host_api_name = p.get_host_api_info_by_index(dev['hostApi'])['name']
        if dev['maxInputChannels'] > 0:
            print(f"ID: {i} | {dev['name']}")
            print(f"   Host API: {host_api_name}")
            print(f"   Входов: {dev['maxInputChannels']} | Частота: {int(dev['defaultSampleRate'])}")

    p.terminate()


def find_asio_steinberg_device() -> Optional[Dict[str, Any]]:
    """
    Принудительно ищет устройство Steinberg внутри Host API 'ASIO'.
    Возвращает информацию об устройстве с максимальным количеством входных каналов.
    """
    p = get_pyaudio_instance()
    asio_idx = -1

    # 1. Находим индекс Host API "ASIO"
    for i in range(p.get_host_api_count()):
        api = p.get_host_api_info_by_index(i)
        if "ASIO" in api['name'].upper():
            asio_idx = i
            print(f"✅ Найден ASIO Host API (индекс: {i})")
            break

    if asio_idx == -1:
        print("❌ Драйвер ASIO не найден в системе PyAudio.")
        print("   Убедитесь, что установлен Yamaha Steinberg USB ASIO driver.")
        p.terminate()
        return None

    # 2. Ищем устройства, принадлежащие этому ASIO API
    steinberg_devices = []

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        # Проверяем, принадлежит ли устройство к ASIO
        if dev['hostApi'] == asio_idx:
            if dev['maxInputChannels'] > 0:
                steinberg_devices.append(dev)
                print(f"   -> Найдено ASIO устройство: {dev['name']} (ID: {i}), Входов: {dev['maxInputChannels']}")

    p.terminate()

    if not steinberg_devices:
        return None

    # 3. Выбираем устройство с наибольшим количеством каналов (обычно это основной вход UR44C)
    # Сортируем по maxInputChannels по убыванию
    best_device = max(steinberg_devices, key=lambda d: d['maxInputChannels'])
    return best_device


def record_audio(
        duration: float = None,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None,
        use_callback: bool = False
) -> np.ndarray:
    if duration is None: duration = getattr(config, 'RECORD_DURATION', 5.0)
    if sample_rate is None: sample_rate = getattr(config, 'SAMPLE_RATE', 96000)

    # КРИТИЧНО: По умолчанию пытаемся записать 8 каналов
    if channels is None:
        channels = getattr(config, 'NUM_CHANNELS', 8)

    p = get_pyaudio_instance()
    target_device = None

    # --- Логика выбора устройства ---
    if device_index is not None:
        try:
            target_device = p.get_device_info_by_index(device_index)
            print(f"🎯 Используется устройство по ID: {device_index} ({target_device['name']})")
        except Exception:
            print(f"❌ Устройство с ID {device_index} не найдено.")
            p.terminate()
            raise
    else:
        # Автоматический поиск ASIO Steinberg
        print("🔍 Поиск устройства Steinberg через ASIO...")
        target_device = find_asio_steinberg_device()

        if not target_device:
            print("❌ Не удалось найти устройство Steinberg через ASIO.")
            print("Попробуйте запустить python main.py --list-devices для диагностики.")
            p.terminate()
            raise ValueError("ASIO Device not found")

    device_id = target_device['index']
    max_hw_channels = target_device['maxInputChannels']

    # --- Корректировка каналов ---
    # Если карта видит только 2 канала в режиме ASIO через PyAudio,
    # возможно, драйвер настроен иначе или PyAudio не видит полный ASIO профиль.
    # Но мы пробуем запросить столько, сколько есть, или максимум 8.

    requested_channels = min(channels, max_hw_channels)

    if requested_channels < 8:
        print(f"⚠️ ВНИМАНИЕ: Устройство сообщает только о {max_hw_channels} входных каналах.")
        print(f"   Ожидалось 8 каналов для полной синхронизации.")
        print(f"   Будет записано каналов: {requested_channels}")
        print(f"   Возможно, потребуется настройка панели управления Yamaha Steinberg USB ASIO.")

    print(f"⚙️ Параметры записи: {requested_channels} кан., {sample_rate} Гц, {duration} сек.")

    frames = []
    audio_format = pyaudio.paInt16

    try:
        stream = p.open(
            format=audio_format,
            channels=requested_channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=getattr(config, 'BUFFER_SIZE', 1024)
        )

        print("🔴 Запись началась...")
        chunks = int(sample_rate / getattr(config, 'BUFFER_SIZE', 1024) * duration)

        for _ in range(chunks):
            data = stream.read(getattr(config, 'BUFFER_SIZE', 1024), exception_on_overflow=False)
            frames.append(data)

        print("⏹️ Запись завершена.")

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        print("   Возможная причина: Устройство занято или не поддерживает запрошенное кол-во каналов/частоту.")
        if 'stream' in locals(): stream.stop_stream()
        p.terminate()
        raise
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

    # Конвертация
    raw_data = b''.join(frames)
    audio_array = np.frombuffer(raw_data, dtype=np.int16)

    # Обрезка до кратного числа каналов
    valid_len = (len(audio_array) // requested_channels) * requested_channels
    audio_array = audio_array[:valid_len].reshape(-1, requested_channels)

    return audio_array.astype(np.float32) / 32768.0


def check_microphone(device_index: Optional[int] = None, duration: float = 2.0):
    # Упрощенная версия проверки с использованием новой логики поиска
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНОВ (ASIO Multi-Channel)")
    print("=" * 60)

    p = get_pyaudio_instance()
    target_device = None

    if device_index is not None:
        target_device = p.get_device_info_by_index(device_index)
    else:
        target_device = find_asio_steinberg_device()

    if not target_device:
        print("❌ Устройство не найдено.")
        p.terminate()
        return {"success": False}

    print(f"Устройство: {target_device['name']}")
    print(f"Доступно каналов: {target_device['maxInputChannels']}")

    # Пробуем записать все доступные каналы
    ch_count = target_device['maxInputChannels']
    sample_rate = int(target_device['defaultSampleRate'])

    # Принудительно пробуем 96кГц если это UR44C
    if "UR44" in target_device['name']:
        sample_rate = 96000

    print(f"Запись {duration} сек на {ch_count} каналов ({sample_rate} Гц)... Говорите во все микрофоны!")

    try:
        data = record_audio(
            duration=duration,
            device_index=target_device['index'],
            channels=ch_count,
            sample_rate=sample_rate
        )

        print(f"\nАнализ {data.shape[1]} каналов:")
        all_silent = True

        for i in range(data.shape[1]):
            channel_data = data[:, i]
            rms = np.sqrt(np.mean(channel_data ** 2))
            db = 20 * np.log10(rms + 1e-10)
            status = "✅ АКТИВЕН" if db > -40 else "❌ ТИШИНА"
            if db > -40: all_silent = False
            print(f"  Канал {i + 1}: {db:.2f} дБ {status}")

        p.terminate()
        return {"success": not all_silent, "channels": data.shape[1]}

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        p.terminate()
        return {"success": False, "error": str(e)}


def list_asio_devices():
    """Обертка для вывода списка."""
    list_all_devices_verbose()
