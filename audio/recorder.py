import sounddevice as sd
import numpy as np
import time
import config
from typing import Optional, Dict, Any, List


def list_asio_devices() -> List[Dict[str, Any]]:
    """
    Возвращает список устройств, которые могут работать в режиме низкой задержки (ASIO/Wasapi).
    Ищем устройства с именем Steinberg.
    """
    devices = []
    print("\n=== ПОИСК УСТРОЙСТВ STEINBERG ===")

    query_devices = sd.query_devices()
    for i, dev in enumerate(query_devices):
        # Ищем устройства Steinberg или те, у которых много входных каналов (>2)
        is_steinberg = "steinberg" in dev['name'].lower() or "ur44" in dev['name'].lower()
        has_many_channels = dev['max_input_channels'] >= 8

        if dev['max_input_channels'] > 0 and (is_steinberg or has_many_channels):
            devices.append({
                'index': i,
                'name': dev['name'],
                'channels': dev['max_input_channels'],
                'hostapi': dev['hostapi'],
                'default_samplerate': dev['default_samplerate']
            })
            print(
                f"ID: {i} | {dev['name']} | Входов: {dev['max_input_channels']} | Частота: {dev['default_samplerate']}")

    return devices


def find_steinberg_device() -> Optional[int]:
    """Находит индекс устройства Steinberg UR44C."""
    devices = list_asio_devices()
    for dev in devices:
        if "steinberg" in dev['name'].lower() or "ur44" in dev['name'].lower():
            # Предпочитаем устройство, которое поддерживает 8 каналов
            if dev['channels'] >= 8:
                return dev['index']
    # Если 8 каналов не найдено явно, берем первое устройство Steinberg
    if devices:
        return devices[0]['index']
    return None


def record_audio(
        duration: float = None,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None
) -> np.ndarray:
    """
    Записывает аудио через sounddevice (использует ASIO драйвер Steinberg).
    """
    if duration is None: duration = config.RECORD_DURATION
    if channels is None: channels = config.NUM_CHANNELS
    if sample_rate is None: sample_rate = config.SAMPLE_RATE

    # Автопоиск устройства, если индекс не передан
    if device_index is None:
        device_index = find_steinberg_device()
        if device_index is None:
            raise ValueError("Не найдено устройство Steinberg UR44C! Запустите --list-asio для проверки.")

    # Проверка возможностей устройства
    dev_info = sd.query_devices(device_index)
    max_ch = dev_info['max_input_channels']

    if channels > max_ch:
        print(f"⚠️ Устройство поддерживает только {max_ch} каналов. Переключаюсь на {max_ch}.")
        channels = max_ch
    else:
        print(f"✅ Запрос {channels} каналов поддержан устройством.")

    print(f"\n🎙️ ЗАПИСЬ ЧЕРЕЗ SOUNDEVICE (ASIO режим)")
    print(f"   Устройство: {dev_info['name']} (ID: {device_index})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")

    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(f"⚠️ Status: {status}")
        frames.append(indata.copy())

    try:
        # Открываем поток с явным указанием устройства и параметров
        with sd.InputStream(
                samplerate=sample_rate,
                device=device_index,
                channels=channels,
                dtype='float32',
                blocksize=config.BUFFER_SIZE,
                callback=callback
        ):
            print("   🟢 Запись началась...")
            sd.sleep(int(duration * 1000))
            print("   🔴 Запись завершена.")

        if not frames:
            raise RuntimeError("Буфер пуст. Проверьте подключение микрофона.")

        audio_data = np.concatenate(frames, axis=0)
        return audio_data

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        print("   Возможно, частота дискретизации занята другим приложением.")
        print("   Попробуйте закрыть браузер, плеер или изменить SAMPLE_RATE в config.py.")
        raise


def check_microphone(device_index: Optional[int] = None, duration: float = 2.0):
    """Проверка микрофона через sounddevice."""
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (SoundDevice + ASIO)")
    print("=" * 60)

    if device_index is None:
        device_index = find_steinberg_device()
        if device_index is None:
            print("❌ Устройство Steinberg не найдено.")
            return {"success": False}

    dev_info = sd.query_devices(device_index)
    print(f"\nУстройство: {dev_info['name']}")
    print(f"Макс. входных каналов: {dev_info['max_input_channels']}")

    # Пробуем записать максимальное количество каналов (до 8)
    test_channels = min(8, dev_info['max_input_channels'])
    sample_rate = config.SAMPLE_RATE

    # Корректировка частоты, если устройство не поддерживает запрошенную
    # (sounddevice обычно делает ресемплинг сам, но лучше быть осторожным)

    print(f"\nТестовая запись ({duration} сек, {test_channels} кан.)... Говорите в микрофон!")

    try:
        data = record_audio(
            duration=duration,
            device_index=device_index,
            channels=test_channels,
            sample_rate=sample_rate
        )

        # Анализируем первый канал (микрофон 1)
        channel_1 = data[:, 0]
        rms = np.sqrt(np.mean(channel_1 ** 2))
        db_level = 20 * np.log10(rms + 1e-10)
        peak = np.max(np.abs(channel_1))
        peak_db = 20 * np.log10(peak + 1e-10)

        print("\n" + "-" * 60)
        print(f"РЕЗУЛЬТАТЫ (Канал 1):")
        print(f"RMS: {db_level:.2f} дБ | Пик: {peak_db:.2f} дБ")

        if db_level > config.MIC_CHECK_THRESHOLD_DB:
            print("✅ МИКРОФОН РАБОТАЕТ!")
            return {"success": True, "db": db_level}
        else:
            print("❌ СИГНАЛ СЛИШКОМ ТИХИЙ. Проверьте Gain и +48V.")
            return {"success": False, "db": db_level}

    except Exception as e:
        print(f"❌ Ошибка теста: {e}")
        return {"success": False, "error": str(e)}
