import pyaudio
import numpy as np
import time
import config
from typing import Optional, Dict, Any, List, Tuple


def get_pyaudio_instance():
    return pyaudio.PyAudio()


def list_all_devices_detailed() -> List[Dict[str, Any]]:
    """
    Возвращает полный список всех устройств PyAudio с подробной информацией.
    """
    p = get_pyaudio_instance()
    devices = []

    print("\n=== ПОЛНЫЙ СКАН УСТРОЙСТВ (PyAudio) ===")

    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            # Нас интересуют только устройства с входами
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate']),
                    'host_api': p.get_host_api_info_by_index(info['hostApi'])['name']
                })
        except Exception:
            continue

    p.terminate()
    return devices


def find_best_device(target_channels: int = 8) -> Optional[Dict[str, Any]]:
    """
    Ищет устройство, которое поддерживает нужное количество каналов.
    Приоритет: Steinberg в названии + макс каналы.
    """
    devices = list_all_devices_detailed()

    if not devices:
        return None

    # 1. Сначала ищем устройство Steinberg с максимальным кол-вом каналов
    steinberg_devices = [d for d in devices if 'steinberg' in d['name'].lower()]

    best_device = None

    # Пытаемся найти среди Steinberg то, у которого каналов >= target_channels
    # Или просто максимум из доступных Steinberg
    if steinberg_devices:
        # Сортируем по количеству каналов (убывание)
        steinberg_devices.sort(key=lambda x: x['channels'], reverse=True)
        best_device = steinberg_devices[0]

        if best_device['channels'] < target_channels:
            print(
                f"⚠️ Внимание: Лучшее устройство Steinberg '{best_device['name']}' имеет только {best_device['channels']} входов.")
            print("   Возможно, драйвер ASIO не активен для Python, и мы видим только стерео-микшер Windows.")
    else:
        # Если Steinberg не найдено явно, ищем любое устройство с 8 каналами
        multi_channel_devices = [d for d in devices if d['channels'] >= target_channels]
        if multi_channel_devices:
            best_device = multi_channel_devices[0]
            print(f"⚠️ Устройство Steinberg не найдено явно. Выбрано многоканальное устройство: {best_device['name']}")
        else:
            # Берем первое попавшееся
            best_device = devices[0]

    return best_device


def record_audio(
        duration: float = None,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None
) -> np.ndarray:
    """
    Запись аудио через PyAudio.
    """
    if duration is None: duration = getattr(config, 'RECORD_DURATION', 5.0)
    if channels is None: channels = getattr(config, 'NUM_CHANNELS', 8)
    if sample_rate is None: sample_rate = getattr(config, 'SAMPLE_RATE', 48000)

    p = get_pyaudio_instance()

    # Выбор устройства
    if device_index is not None:
        try:
            dev_info = p.get_device_info_by_index(device_index)
        except IOError:
            raise ValueError(f"Устройство с индексом {device_index} не найдено")
    else:
        dev_info = find_best_device(channels)
        if not dev_info:
            raise ValueError("Подходящее аудиоустройство не найдено")

    device_idx = dev_info['index']
    max_ch = dev_info['channels']

    # Корректировка каналов
    if channels > max_ch:
        print(f"⚠️ Запрошено {channels} каналов, но устройство поддерживает только {max_ch}.")
        print(f"   Переключаюсь на {max_ch} каналов.")
        channels = max_ch

    print(f"🎙️ Запись: Устройство '{dev_info['name']}' (ID: {device_idx})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
    print(f"   Host API: {dev_info['host_api']}")

    frames = []
    audio_format = pyaudio.paInt16
    chunk_size = 1024

    stream = None
    try:
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=chunk_size
        )

        print("   🟢 Запись началась...")
        for _ in range(int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)
        print("   🔴 Запись завершена.")

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        if stream: stream.stop_stream()
        p.terminate()
        raise

    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

    # Конвертация в numpy
    raw_data = b''.join(frames)
    audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_array.reshape(-1, channels)


def check_microphone(device_index: Optional[int] = None, duration: float = 2.0):
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio)")
    print("=" * 60)

    p = get_pyaudio_instance()

    if device_index is not None:
        try:
            dev_info = p.get_device_info_by_index(device_index)
        except:
            print(f"❌ Устройство {device_index} не найдено")
            return {"success": False}
    else:
        dev_info = find_best_device()
        if not dev_info:
            print("❌ Устройства не найдены")
            return {"success": False}

    print(f"\nУстройство: {dev_info['name']}")
    print(f"Доступно входов: {dev_info['channels']}")

    # Пробуем записать столько каналов, сколько дает устройство (макс 8)
    rec_channels = min(dev_info['channels'], 8)
    sample_rate = int(dev_info['sample_rate'])

    print(f"Тестовая запись ({duration} сек) на {rec_channels} кан./{sample_rate} Гц...")
    print("Говорите в микрофон!")

    try:
        data = record_audio(
            duration=duration,
            device_index=dev_info['index'],
            channels=rec_channels,
            sample_rate=sample_rate
        )

        # Анализируем первый канал
        channel_0 = data[:, 0] if data.ndim > 1 else data
        rms = np.sqrt(np.mean(channel_0 ** 2))
        db = 20 * np.log10(rms + 1e-10)

        print("\n" + "-" * 40)
        print(f"Уровень сигнала (RMS): {db:.2f} дБ")

        if db > -40:
            print("✅ МИКРОФОН РАБОТАЕТ!")
            return {"success": True, "db": db, "channels": rec_channels}
        else:
            print("❌ ТИШИНА. Проверьте подключение и Gain.")
            return {"success": False, "db": db}

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return {"success": False}
    finally:
        p.terminate()
