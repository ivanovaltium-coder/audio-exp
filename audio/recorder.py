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
        device_index: int = 53,  # По умолчанию ваш ID
        channels: int = 8,
        sample_rate: int = 96000,
        output_filename: str = "recording_8ch.wav",
        save_file: bool = True
) -> np.ndarray:
    if duration is None:
        duration = getattr(config, 'RECORD_DURATION', 5.0)

    p = pyaudio.PyAudio()

    # Формат 24 бита
    audio_format = pyaudio.paInt24

    print(f"🎙️ Запись: Устройство 'Yamaha Steinberg USB ASIO' (ID: {device_index})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
    print(f"   Формат: 24 бита (paInt24)")

    frames = []

    try:
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=4096
        )

        print("   🟢 Запись началась...")
        chunks_to_record = int(sample_rate / 4096 * duration)

        for _ in range(chunks_to_record):
            data = stream.read(4096, exception_on_overflow=False)
            frames.append(data)

        print("   🔴 Запись завершена.")
        stream.stop_stream()
        stream.close()

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        if 'stream' in locals():
            stream.close()
        p.terminate()
        raise

    p.terminate()

    # Конвертация в numpy массив для обработки
    # Для 24 бит нужна аккуратная обработка байтов
    raw_data = b''.join(frames)

    # Преобразование 24 бит (3 байта) в 32 бита (4 байта) для numpy
    # Добавляем нулевой байт в начало каждого сэмпла (знаковый int32)
    samples = []
    sample_size = 3 * channels
    num_samples = len(raw_data) // sample_size

    # Быстрая векторизованная конвертация
    buffer = np.frombuffer(raw_data, dtype=np.uint8)
    buffer = buffer.reshape(-1, 3 * channels)

    # Создаем массив int32
    audio_int32 = np.zeros((buffer.shape[0], channels), dtype=np.int32)

    for ch in range(channels):
        # Сдвиг байтов для каждого канала
        # Байты идут подряд: [ch0_b0, ch0_b1, ch0_b2, ch1_b0, ch1_b1, ch1_b2, ...]
        # Нам нужно собрать их в int32 со знаком
        offset = ch * 3
        b0 = buffer[:, offset].astype(np.int32)
        b1 = buffer[:, offset + 1].astype(np.int32)
        b2 = buffer[:, offset + 2].astype(np.int32)

        # Сборка 24 бит в 32 бит (с учетом знака)
        val = (b2 << 16) | (b1 << 8) | b0
        val = np.where(val >= 0x800000, val - 0x1000000, val)  # Знаковое расширение
        audio_int32[:, ch] = val

    # Нормализация в float32 [-1, 1]
    audio_float = audio_int32.astype(np.float32) / 8388608.0  # 2^23

    print(f"✅ Записано: {audio_float.shape}")

    # Сохранение в WAV файл
    if save_file:
        try:
            import wave
            # Для записи 24 бит в wave нужно использовать особый подход,
            # но стандартный wave модуль плохо дружит с 24 битами напрямую из numpy.
            # Проще сохранить как 32 бита (pcms32) или 16 бит для совместимости.
            # Сохраним как 32 бита (ближе всего к оригиналу)

            wf = wave.open(output_filename, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(4)  # 4 байта = 32 бита
            wf.setframerate(sample_rate)

            # Конвертируем float32 обратно в int32 для записи
            audio_save = (audio_float * 2147483647).astype(np.int32)
            wf.writeframes(audio_save.tobytes())
            wf.close()
            print(f"💾 Файл сохранен: {os.path.abspath(output_filename)}")
        except Exception as e:
            print(f"⚠️ Не удалось сохранить WAV: {e}")

    return audio_float

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
