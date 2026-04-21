import pyaudio
import wave
import numpy as np
import time
import os
from typing import Optional, Dict, Any, List

# Импорт конфигурации
try:
    import config
except ImportError:
    # Если запускается вне проекта, задаем значения по умолчанию
    class Config:
        SAMPLE_RATE = 96000
        NUM_CHANNELS = 8
        RECORD_DURATION = 5.0
        BUFFER_SIZE = 4096
        AUDIO_FORMAT = pyaudio.paInt24
        MIC_CHECK_THRESHOLD_DB = -40
        ASIO_DEVICE_NAME = "Steinberg"


    config = Config()


def get_pyaudio_instance() -> pyaudio.PyAudio:
    """Создает и возвращает экземпляр PyAudio."""
    return pyaudio.PyAudio()


def get_asio_device_id(device_name: str = None) -> Optional[int]:
    """
    Ищет устройство ASIO (Steinberg) и возвращает его ID.
    """
    p = get_pyaudio_instance()
    target_id = None

    print("\n=== ПОИСК УСТРОЙСТВА ЧЕРЕЗ ASIO API ===")

    # 1. Находим индекс Host API ASIO
    asio_api_index = -1
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_index = i
            break

    if asio_api_index == -1:
        print("❌ Ошибка поиска: ❌ Драйвер ASIO не найден в системе PyAudio!")
        print("Убедитесь, что установлен Yamaha Steinberg USB ASIO и закрыт в других программах.")
        p.terminate()
        return None

    # 2. Ищем устройства внутри ASIO API
    found_devices = []
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['hostApi'] == asio_api_index:
            if dev_info['maxInputChannels'] > 0:
                found_devices.append(dev_info)

    if not found_devices:
        print("❌ Входов ASIO не найдено.")
        p.terminate()
        return None

    # 3. Выбираем нужное устройство
    # Если имя задано, ищем совпадение, иначе берем первое с макс каналами
    target_dev = None
    if device_name:
        for dev in found_devices:
            if device_name.lower() in dev['name'].lower():
                target_dev = dev
                break

    if not target_dev and found_devices:
        # Берем устройство с наибольшим числом входов (обычно это сама карта)
        target_dev = max(found_devices, key=lambda d: d['maxInputChannels'])

    if target_dev:
        target_id = target_dev['index']
        print(f"✅ Найдено ASIO устройство: {target_dev['name']} (ID: {target_id})")
        print(f"   Входов: {target_dev['maxInputChannels']}, Частота: {target_dev['defaultSampleRate']}")
    else:
        print("❌ Подходящее устройство не найдено.")

    p.terminate()
    return target_id


def record_audio(
        duration: float = None,
        device_index: int = None,
        channels: int = None,
        sample_rate: int = None,
        output_filename: str = "recording_8ch.wav"
) -> Optional[np.ndarray]:
    """
    Записывает аудио через ASIO драйвер.
    Возвращает numpy массив и сохраняет WAV файл.
    """
    # Значения по умолчанию из config
    if duration is None: duration = getattr(config, 'RECORD_DURATION', 5.0)
    if channels is None: channels = getattr(config, 'NUM_CHANNELS', 8)
    if sample_rate is None: sample_rate = getattr(config, 'SAMPLE_RATE', 96000)

    audio_format = getattr(config, 'AUDIO_FORMAT', pyaudio.paInt24)
    chunk_size = getattr(config, 'BUFFER_SIZE', 4096)

    p = get_pyaudio_instance()

    # Определение устройства
    if device_index is None:
        device_index = get_asio_device_id(getattr(config, 'ASIO_DEVICE_NAME', "Steinberg"))
        if device_index is None:
            p.terminate()
            raise ValueError("ASIO устройство не найдено")

    try:
        dev_info = p.get_device_info_by_index(device_index)
        dev_name = dev_info['name']
    except Exception:
        dev_name = f"Device #{device_index}"

    print(f"🎙️ Запись: Устройство '{dev_name}' (ID: {device_index})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
    print(f"   Формат: {'24 бита (paInt24)' if audio_format == pyaudio.paInt24 else '16 бит (paInt16)'}")

    stream = None
    frames = []

    try:
        # Открываем поток
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )

        print("   🟢 Запись началась...")
        num_chunks = int(sample_rate / chunk_size * duration)

        for _ in range(num_chunks):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(data)

        print("   🔴 Запись завершена.")

    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        raise
    finally:
        # Корректное закрытие потока
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        p.terminate()

    # Обработка данных
    raw_data = b''.join(frames)

    # Конвертация в numpy в зависимости от формата
    if audio_format == pyaudio.paInt24:
        # 24 бита (3 байта) -> конвертируем в int32 (4 байта) для удобства
        # Добавляем нулевой байт в начало каждого сэмпла (знаковое расширение)
        samples = np.frombuffer(raw_data, dtype=np.uint8)
        # Reshape: (кол-во сэмплов * 3) -> (кол-во сэмплов, 3)
        samples = samples.reshape(-1, 3)

        # Преобразование 3 байт в 4 байта (int32) со знаковым расширением
        # Байты идут в порядке Little Endian: [B0, B1, B2] -> [B0, B1, B2, Sign(B2)]
        padded_samples = np.zeros((samples.shape[0], 4), dtype=np.uint8)
        padded_samples[:, :3] = samples

        # Знаковый байт (третий байт) определяет знак.
        # Если старший бит 3-го байта установлен (>=128), то число отрицательное.
        # Заполняем 4-й байт единицами (0xFF), если число отрицательное.
        mask = (samples[:, 2] & 0x80) != 0
        padded_samples[mask, 3] = 0xFF

        # Теперь интерпретируем как int32
        audio_array = padded_samples.view(dtype=np.int32).reshape(-1, channels)

        # Нормализация в float32 [-1, 1] (для 24 бит делим на 2^23)
        audio_float = audio_array.astype(np.float32) / 8388608.0

    elif audio_format == pyaudio.paInt16:
        audio_array = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, channels)
        audio_float = audio_array.astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Неподдерживаемый формат: {audio_format}")

    # Сохранение в WAV файл
    save_path = os.path.abspath(output_filename)
    try:
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(3 if audio_format == pyaudio.paInt24 else 2)  # 3 байта для 24 бит
        wf.setframerate(sample_rate)
        wf.writeframes(raw_data)
        wf.close()
        print(f"💾 Файл сохранен: {save_path}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить файл: {e}")

    return audio_float


def check_microphone(
        device_index: int = None,
        duration: float = 2.0
) -> Dict[str, Any]:
    """Проверка работы микрофона."""
    print("=" * 60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio + ASIO)")
    print("=" * 60)

    try:
        # Записываем данные (используем дефолтные настройки из config: 8 каналов, 96кГц)
        audio_data = record_audio(
            duration=duration,
            device_index=device_index,
            output_filename="test_check.wav"  # Временный файл
        )

        if audio_data is None or audio_data.size == 0:
            print("❌ Нет данных для анализа.")
            return {"success": False}

        # Анализируем первый канал
        channel_0 = audio_data[:, 0]
        rms = np.sqrt(np.mean(channel_0 ** 2))
        db_level = 20 * np.log10(rms + 1e-10)
        peak = np.max(np.abs(channel_0))

        threshold = getattr(config, 'MIC_CHECK_THRESHOLD_DB', -40)

        print("\n" + "-" * 60)
        print(f"РЕЗУЛЬТАТЫ (Канал 1 из {audio_data.shape[1]}):")
        print("-" * 60)
        print(f"Средний уровень (RMS): {db_level:.2f} дБ")
        print(f"Пиковый уровень: {20 * np.log10(peak + 1e-10):.2f} дБ")

        success = db_level > threshold

        if success:
            print(f"\n✅ МИКРОФОН РАБОТАЕТ! (Сигнал > порога {threshold} дБ)")
        else:
            print(f"\n❌ СИГНАЛ СЛИШКОМ ТИХИЙ (< {threshold} дБ)")
            print("Проверьте подключение, фантомное питание +48V и ручки Gain.")

        # Удаляем тестовый файл
        if os.path.exists("test_check.wav"):
            os.remove("test_check.wav")

        return {"success": success, "rms_db": db_level}

    except Exception as e:
        print(f"\n❌ Ошибка проверки: {e}")
        return {"success": False, "error": str(e)}
