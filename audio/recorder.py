import pyaudio
import wave
import numpy as np
import time
import os
import sys
from typing import Optional, Tuple, List, Dict, Any
import config


def get_pyaudio_instance():
    """Создает и возвращает экземпляр PyAudio."""
    return pyaudio.PyAudio()


def list_asio_devices() -> List[Dict[str, Any]]:
    """
    Возвращает список всех устройств, доступных в системе через PyAudio.
    Фильтрует только те, у которых есть входные каналы.
    """
    p = get_pyaudio_instance()
    devices = []

    print("Сканирование всех устройств PyAudio...")

    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            # Нам нужны только устройства с входными каналами
            if dev_info['maxInputChannels'] > 0:
                # Фильтруем по имени, если нужно искать именно Steinberg,
                # но пока вернем все, чтобы пользователь выбрал
                devices.append({
                    'index': i,
                    'name': dev_info['name'],
                    'channels': dev_info['maxInputChannels'],
                    'default_sample_rate': int(dev_info['defaultSampleRate']),
                    'hostApi': dev_info['hostApi']
                })
        except Exception:
            continue

    p.terminate()
    return devices


def find_asio_device(device_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ищет устройство по имени или возвращает первое подходящее Steinberg.
    """
    all_devices = list_asio_devices()

    if not all_devices:
        return None

    # Если имя задано, ищем точное совпадение
    if device_name:
        for dev in all_devices:
            if device_name.lower() in dev['name'].lower():
                return dev

    # Если имя не задано или не найдено, ищем первое устройство Steinberg
    for dev in all_devices:
        if "steinberg" in dev['name'].lower() or "ur44" in dev['name'].lower():
            return dev

    # Если и этого нет, возвращаем первое доступное
    return all_devices[0] if all_devices else None


def record_audio(
        duration: float = None,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None,
        use_callback: bool = False
) -> np.ndarray:
    """
    Записывает аудио с использованием PyAudio.
    КРИТИЧНО: Для Steinberg UR44C по умолчанию используем 8 каналов и 96000 Гц,
    чтобы задействовать полный потенциал драйвера.
    """
    # Значения по умолчанию из config или жестко заданные для UR44C
    if duration is None:
        duration = getattr(config, 'RECORD_DURATION', 5.0)
    if channels is None:
        # ВАЖНО: Пробуем взять 8 каналов (полный режим карты)
        channels = getattr(config, 'NUM_CHANNELS', 8)
    if sample_rate is None:
        # ВАЖНО: 96 кГц для высокого качества и синхронизации
        sample_rate = getattr(config, 'SAMPLE_RATE', 96000)

    p = get_pyaudio_instance()

    # --- Определение устройства ---
    target_device = None
    if device_index is not None:
        try:
            target_device = p.get_device_info_by_index(device_index)
        except IOError:
            print(f"❌ Устройство с индексом {device_index} не найдено.")
            p.terminate()
            raise ValueError(f"Device {device_index} not found")
    else:
        # Пытаемся найти устройство автоматически
        target_device = find_asio_device(getattr(config, 'ASIO_DEVICE_NAME', "Steinberg"))
        if not target_device:
            print("❌ Не найдено подходящее устройство Steinberg.")
            print("Запустите 'python main.py --list-asio' для просмотра доступных устройств.")
            p.terminate()
            raise ValueError("No suitable audio device found")

    device_idx = target_device['index']
    # Корректное получение количества каналов (ключ может называться по-разному в разных контекстах, но здесь стандарт PyAudio)
    available_channels = target_device.get('maxInputChannels', target_device.get('channels', 0))

    # --- Корректировка параметров ---
    # Если запрошено больше каналов, чем доступно физически на этом конкретном входе
    if channels > available_channels:
        print(
            f"⚠️ Запрошено {channels} каналов, но устройство '{target_device['name']}' поддерживает только {available_channels}.")
        print(f"   Переключаюсь на {available_channels} каналов.")
        channels = available_channels

    # Убеждаемся, что частота поддерживается (PyAudio обычно сам делает ресемплинг, но лучше предупредить)
    # Для ASIO драйверов часто можно задать любую частоту, и драйвер переключит карту

    print(f"🎙️ Запись: Устройство '{target_device['name']}' (ID: {device_idx})")
    print(
        f"   Каналы: {channels} (из макс. {available_channels}), Частота: {sample_rate} Гц, Длительность: {duration} сек")

    frames = []
    audio_format = pyaudio.paInt16  # Используем 16 бит для надежности и скорости

    try:
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=getattr(config, 'BUFFER_SIZE', 1024),
            stream_callback=None  # Пока используем polling для простоты записи в файл
        )

        print("   🟢 Запись началась...")
        chunks_to_record = int(sample_rate / getattr(config, 'BUFFER_SIZE', 1024) * duration)

        for _ in range(chunks_to_record):
            data = stream.read(getattr(config, 'BUFFER_SIZE', 1024), exception_on_overflow=False)
            frames.append(data)

        print("   🔴 Запись завершена.")

    except Exception as e:
        print(f"❌ Ошибка во время записи: {e}")
        print("   Возможно, устройство занято другой программой или не поддерживает выбранную частоту/каналы.")
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        raise
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

    # --- Конвертация в Numpy ---
    raw_data = b''.join(frames)
    audio_array = np.frombuffer(raw_data, dtype=np.int16)

    # Reshape в (samples, channels)
    # Если данных меньше, чем ожидалось (обрыв), обрезаем до кратного числа
    total_samples = len(audio_array) // channels
    audio_array = audio_array[:total_samples * channels]

    audio_array = audio_array.reshape(-1, channels)

    # Нормализация в float32 [-1, 1]
    return audio_array.astype(np.float32) / 32768.0


def check_microphone(
        device_index: Optional[int] = None,
        duration: float = 2.0
) -> Dict[str, Any]:
    """
    Проверяет, работает ли микрофон.
    """
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio + ASIO)")
    print("=" * 60)

    p = get_pyaudio_instance()

    # Поиск устройства
    target_device = None
    if device_index is not None:
        try:
            target_device = p.get_device_info_by_index(device_index)
        except IOError:
            print(f"❌ Устройство {device_index} не найдено.")
            p.terminate()
            return {"success": False, "error": "Device not found"}
    else:
        target_device = find_asio_device(getattr(config, 'ASIO_DEVICE_NAME', "Steinberg"))
        if not target_device:
            print("❌ Устройства не найдены.")
            p.terminate()
            return {"success": False, "error": "No devices found"}

    print(f"\nИспользуется устройство: {target_device['name']}")
    # Безопасное получение каналов
    avail_ch = target_device.get('maxInputChannels', target_device.get('channels', 0))
    print(f"Доступно каналов на устройстве: {avail_ch}")

    # Для теста записываем столько каналов, сколько доступно, но анализируем первый
    test_channels = min(avail_ch, 8)  # Не больше 8
    sample_rate = getattr(config, 'SAMPLE_RATE', 96000)

    # Попытка установить частоту устройства. Если устройство не поддерживает, PyAudio может выдать ошибку.
    # Попробуем использовать родную частоту устройства, если 96к не пойдет
    try:
        test_rate = int(target_device['defaultSampleRate'])
        # Если хотим принудительно 96к, раскомментируйте строку ниже, но это может вызвать ошибку, если карта занята в другом режиме
        # test_rate = 96000
        print(f"Частота дискретизации: {test_rate} Гц")
    except KeyError:
        test_rate = 44100

    print(f"\nЗапись {duration} сек... (говорите в микрофон)")

    try:
        audio_data = record_audio(
            duration=duration,
            device_index=target_device['index'],
            channels=test_channels,
            sample_rate=test_rate,
            use_callback=False
        )

        # Анализ первого канала (индекс 0)
        if audio_data.ndim > 1:
            channel_0 = audio_data[:, 0]
        else:
            channel_0 = audio_data

        rms = np.sqrt(np.mean(channel_0 ** 2))
        db_level = 20 * np.log10(rms + 1e-10)

        peak = np.max(np.abs(channel_0))
        peak_db = 20 * np.log10(peak + 1e-10)

        threshold_db = getattr(config, 'MIC_CHECK_THRESHOLD_DB', -40)

        print("\n" + "-" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА (Канал 1):")
        print("-" * 60)
        print(f"Средний уровень (RMS): {db_level:.2f} дБ")
        print(f"Пиковый уровень: {peak_db:.2f} дБ")
        print(f"Порог обнаружения сигнала: {threshold_db} дБ")

        success = db_level > threshold_db

        if success:
            print("\n✅ МИКРОФОН РАБОТАЕТ!")
            print(f"   Сигнал обнаружен (уровень {db_level:.1f} дБ > порога {threshold_db} дБ)")
            print("   Запись идёт с конденсаторного микрофона через звуковую карту.")
        else:
            print("\n❌ МИКРОФОН НЕ РАБОТАЕТ ИЛИ СИГНАЛ СЛИШКОМ ТИХИЙ")
            print(f"   Уровень сигнала {db_level:.1f} дБ < порога {threshold_db} дБ")
            print("\nРекомендации:")
            print("  1. Проверьте подключение микрофона к входам 1-4 (XLR) на задней панели.")
            print("  2. Включите фантомное питание +48V на передней панели UR44C.")
            print("  3. Увеличьте ручку GAIN для соответствующего канала на верхней панели.")
            print("  4. Убедитесь, что в Windows выбран режим работы ASIO (не MME/Wasapi).")

        p.terminate()
        return {
            "success": success,
            "rms_db": db_level,
            "peak_db": peak_db,
            "device_name": target_device['name'],
            "channels_recorded": test_channels
        }

    except Exception as e:
        print(f"\n❌ Ошибка при проверке: {e}")
        print("   Попробуйте изменить частоту дискретизации в config.py или закрыть другие аудио-приложения.")
        p.terminate()
        return {"success": False, "error": str(e)}
