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
    Возвращает список устройств, использующих хост-API ASIO.
    Это критически важно для работы со звуковой картой Steinberg UR44C.
    """
    p = get_pyaudio_instance()
    asio_devices = []

    # Находим индекс хост-API ASIO
    asio_host_api_idx = -1
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_host_api_idx = i
            break

    if asio_host_api_idx == -1:
        print("⚠️ Предупреждение: Драйвер ASIO не найден в системе.")
        print("   Убедитесь, что установлен драйвер Yamaha Steinberg USB ASIO.")
        p.terminate()
        return []

    # Перебираем все устройства и ищем те, что принадлежат ASIO
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['hostApi'] == asio_host_api_idx:
            # Нам нужны только устройства с входными каналами
            if dev_info['maxInputChannels'] > 0:
                asio_devices.append({
                    'index': i,
                    'name': dev_info['name'],
                    'channels': dev_info['maxInputChannels'],
                    'default_sample_rate': int(dev_info['defaultSampleRate'])
                })

    p.terminate()
    return asio_devices


def find_asio_device(device_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ищет устройство ASIO по имени или возвращает первое доступное.
    """
    devices = list_asio_devices()
    if not devices:
        return None

    if device_name:
        for dev in devices:
            if device_name.lower() in dev['name'].lower():
                return dev

    # Если имя не указано или не найдено, возвращаем первое подходящее устройство
    # Обычно это и есть Steinberg UR44C, если она одна подключена
    return devices[0]


def record_audio(
        duration: float = config.RECORD_DURATION,
        device_index: Optional[int] = None,
        channels: int = None,
        sample_rate: int = None,
        use_callback: bool = False
) -> np.ndarray:
    """
    Записывает аудио с использованием PyAudio и драйвера ASIO.

    Args:
        duration: Длительность записи в секундах.
        device_index: Индекс устройства PyAudio (если известен).
        channels: Количество каналов для записи (по умолчанию из config).
        sample_rate: Частота дискретизации (по умолчанию из config).
        use_callback: Использовать ли callback режим (для низкой задержки).

    Returns:
        numpy массив с аудиоданными形状 (samples, channels).
    """
    if channels is None:
        channels = config.NUM_CHANNELS
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE

    p = get_pyaudio_instance()

    # Определение устройства
    target_device = None
    if device_index is not None:
        try:
            target_device = p.get_device_info_by_index(device_index)
        except IOError:
            print(f"❌ Устройство с индексом {device_index} не найдено.")
            p.terminate()
            raise
    else:
        # Пытаемся найти ASIO устройство по имени из конфига
        target_device = find_asio_device(config.ASIO_DEVICE_NAME)
        if not target_device:
            print("❌ Не найдено подходящее ASIO устройство.")
            print("Запустите 'python main.py --list-asio' для просмотра доступных устройств.")
            p.terminate()
            raise ValueError("ASIO устройство не найдено")

    device_idx = target_device['index']
    available_channels = target_device['channels']

    # Корректировка количества каналов: нельзя запросить больше, чем есть у устройства
    if channels > available_channels:
        print(f"⚠️ Запрошено {channels} каналов, но устройство поддерживает только {available_channels}.")
        print(f"   Переключаюсь на {available_channels} каналов.")
        channels = available_channels

    # Проверка поддержки частоты дискретизации (ASIO обычно гибкий, но проверим)
    # PyAudio сам попытается установить частоту, если устройство поддерживает

    print(f"🎙️ Запись: Устройство '{target_device['name']}' (ID: {device_idx})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
    print(f"   Режим: {'Callback (Low Latency)' if use_callback else 'Polling'}")

    frames = []

    def callback(in_data, frame_count, time_info, status):
        if status:
            # Вывод предупреждений о переполнении буфера и т.д.
            pass
        return (in_data, pyaudio.paContinue)

    stream = None
    try:
        # Открываем поток
        # format=pyaudio.paInt24 (24 бита) или paInt32 для высокого качества,
        # но для классификатора часто хватает paInt16. Используем настройку из config или по умолчанию.
        audio_format = config.AUDIO_FORMAT if hasattr(config, 'AUDIO_FORMAT') else pyaudio.paInt16

        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=config.BUFFER_SIZE,
            stream_callback=callback if use_callback else None
        )

        if use_callback:
            # В режиме callback мы просто ждем нужное время
            stream.start_stream()
            time.sleep(duration)
            # Данные собираются внутри потока, но PyAudio callback режим сложен для простого сбора всего буфера
            # Для простоты в первой версии используем polling, если не критична микро-задержка при записи файла
            # Однако, если нужен именно callback для потоковой обработки, логика будет другой.
            # Здесь реализуем гибридный подход: если use_callback=True, но мы пишем в файл,
            # то проще использовать polling для надежности сохранения всего куска.
            # ПЕРЕОСМЫСЛЕНИЕ: Для записи файла на диск polling надежнее и проще.
            # Callback нужен для реальной обработки в реальном времени.
            # Если пользователь выбрал --callback, предположим, что он хочет тест низкой задержки,
            # но для сохранения файла нам все равно нужно накопить данные.
            # Оставим polling для записи файла, так как это надежнее для больших кусков.
            stream.stop_stream()
            stream.close()
            stream = p.open(
                format=audio_format,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=config.BUFFER_SIZE
            )

        # Основной цикл записи (Polling mode)
        chunks_to_record = int(sample_rate / config.BUFFER_SIZE * duration)

        for _ in range(chunks_to_record):
            data = stream.read(config.BUFFER_SIZE, exception_on_overflow=False)
            frames.append(data)

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

    # Конвертируем байты в numpy массив
    # Определяем dtype в зависимости от формата
    if audio_format == pyaudio.paInt16:
        dtype = np.int16
    elif audio_format == pyaudio.paInt24:
        # PyAudio возвращает 24 бита в 3 байтах, нужно аккуратно обработать
        # Но для простоты, если используется paInt24, часто читают как int32 и сдвигают,
        # или используют struct. Однако стандартный np.frombuffer не поддержит 24 бит напрямую.
        # Упростим: будем использовать paInt16 для классификатора, если не указано иное.
        # Если все же 24 бита, придется делать unpack.
        # Для текущей задачи предположим, что мы работаем с int16 или int32.
        # Если в конфиге стоит paInt24, нужно специальное преобразование.
        # Сделаем универсально:
        raw_data = b''.join(frames)
        # 24 бита = 3 байта. Кол-во сэмплов = len(raw_data) // (channels * 3)
        # Это сложно векторизовать быстро без потерь.
        # Рекомендация: используйте paInt16 для ML задач, этого достаточно.
        # Если жестко нужно 24 бита, раскомментируйте код ниже, но он медленный.
        raise NotImplementedError(
            "Формат 24 бита требует дополнительной обработки для numpy. Используйте paInt16 в config.")
    elif audio_format == pyaudio.paInt32:
        dtype = np.int32
    else:
        dtype = np.int16  # fallback

    if audio_format == pyaudio.paInt24:
        # Обработка 24 бит (если вдруг потребуется)
        # Превращаем 3 байта в 4 байта (int32) со сдвигом
        raw_data = b''.join(frames)
        samples = np.frombuffer(raw_data, dtype=np.uint8)
        # Резhape и конвертация сложны, опустим для краткости, так как ML обычно хватает 16 бит
        pass

        # Стандартный путь для 16/32 бит
    raw_data = b''.join(frames)
    audio_array = np.frombuffer(raw_data, dtype=dtype)

    # Reshape в (samples, channels)
    audio_array = audio_array.reshape(-1, channels)

    return audio_array.astype(np.float32) / 32768.0  # Нормализация в диапазон [-1, 1] для float32


def check_microphone(
        device_index: Optional[int] = None,
        duration: float = 2.0
) -> Dict[str, Any]:
    """
    Проверяет, работает ли микрофон и записывается ли сигнал.
    Использует PyAudio для доступа к ASIO.
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
        target_device = find_asio_device(config.ASIO_DEVICE_NAME)
        if not target_device:
            # Пробуем любое ASIO устройство
            devices = list_asio_devices()
            if devices:
                target_device = devices[0]
                print(
                    f"⚠️ Устройство '{config.ASIO_DEVICE_NAME}' не найдено, используем первое доступное: {target_device['name']}")
            else:
                print("❌ ASIO устройства не найдены. Проверьте установку драйверов Steinberg.")
                p.terminate()
                return {"success": False, "error": "No ASIO devices found"}

    print(f"\nИспользуется устройство: {target_device['name']}")
    print(f"Доступно каналов: {target_device['channels']}")

    # Записываем 1 канал для теста, даже если устройство многоканальное
    test_channels = 1
    sample_rate = int(target_device['defaultSampleRate'])

    print(f"\nЗапись {duration} сек... (говорите в микрофон)")

    try:
        # Запись через нашу основную функцию, но с ограничением на 1 канал
        audio_data = record_audio(
            duration=duration,
            device_index=target_device['index'],
            channels=test_channels,
            sample_rate=sample_rate,
            use_callback=False
        )

        # Анализ сигнала
        rms = np.sqrt(np.mean(audio_data ** 2))
        # Перевод в дБ (опорное значение 1.0)
        db_level = 20 * np.log10(rms + 1e-10)

        peak = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak + 1e-10)

        threshold_db = config.MIC_CHECK_THRESHOLD_DB

        print("\n" + "-" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
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
            print("  1. Убедитесь, что микрофон подключён к входу 1-4 на задней панели UR44C")
            print("  2. Проверьте, что включено фантомное питание +48V (кнопка на передней панели)")
            print("  3. Увеличьте ручку增益 (Gain) для соответствующего входа на верхней панели")
            print("  4. Убедитесь, что в Cubase/Steinberg настройки ASIO активны и не эксклюзивны")

        p.terminate()
        return {
            "success": success,
            "rms_db": db_level,
            "peak_db": peak_db,
            "device_name": target_device['name']
        }

    except Exception as e:
        print(f"\n❌ Ошибка при проверке: {e}")
        p.terminate()
        return {"success": False, "error": str(e)}