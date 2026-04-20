# audio/recorder.py
"""
Модуль записи аудио с поддержкой ASIO драйвера Steinberg UR44C через PyAudio.
Обеспечивает низкоуровневую запись с минимальной задержкой для точного фазового анализа.
Критически важно использовать именно PyAudio с официальным .whl для работы с ASIO на Windows.
"""
import pyaudio
import numpy as np
import config
import time
import sys
from typing import Optional, List, Dict, Any


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
    Возвращает dict с информацией об устройстве или None.
    """
    devices = list_asio_devices()
    if not devices:
        return None
    
    if device_name:
        for dev in devices:
            if device_name.lower() in dev['name'].lower():
                return dev
    
    # Если имя не указано или не найдено, возвращаем первое подходящее устройство
    return devices[0]


def record_audio(
    duration: float = None,
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
        numpy массив с аудиоданными формы (samples, channels).
    """
    if duration is None:
        duration = config.RECORD_DURATION if hasattr(config, 'RECORD_DURATION') else config.WINDOW_SEC
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
        target_device = find_asio_device(config.ASIO_DEVICE_NAME if hasattr(config, 'ASIO_DEVICE_NAME') else None)
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
    
    print(f"🎙️ Запись: Устройство '{target_device['name']}' (ID: {device_idx})")
    print(f"   Каналы: {channels}, Частота: {sample_rate} Гц, Длительность: {duration} сек")
    
    frames = []
    
    try:
        # Используем paInt16 для совместимости с ML моделями
        audio_format = pyaudio.paInt16
        
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_idx,
            frames_per_buffer=config.BUFFER_SIZE if hasattr(config, 'BUFFER_SIZE') else 1024
        )
        
        # Основной цикл записи
        chunks_to_record = int(sample_rate / (config.BUFFER_SIZE if hasattr(config, 'BUFFER_SIZE') else 1024) * duration)
        
        print(f"   Запись... (говорите в микрофон)")
        for _ in range(chunks_to_record):
            data = stream.read(config.BUFFER_SIZE if hasattr(config, 'BUFFER_SIZE') else 1024, exception_on_overflow=False)
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
    raw_data = b''.join(frames)
    audio_array = np.frombuffer(raw_data, dtype=np.int16)
    
    # Reshape в (samples, channels)
    audio_array = audio_array.reshape(-1, channels)
    
    # Нормализация в диапазон [-1, 1] для float32
    return audio_array.astype(np.float32) / 32768.0


def check_microphone(
    device_index: Optional[int] = None,
    duration: float = 2.0
) -> Dict[str, Any]:
    """
    Проверяет, работает ли микрофон и записывается ли сигнал.
    Использует PyAudio для доступа к ASIO.
    """
    print("="*60)
    print("ПРОВЕРКА МИКРОФОНА (PyAudio + ASIO)")
    print("="*60)
    
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
        target_device = find_asio_device(config.ASIO_DEVICE_NAME if hasattr(config, 'ASIO_DEVICE_NAME') else None)
        if not target_device:
            # Пробуем любое ASIO устройство
            devices = list_asio_devices()
            if devices:
                target_device = devices[0]
                print(f"⚠️ Устройство по умолчанию не найдено, используем: {target_device['name']}")
            else:
                print("❌ ASIO устройства не найдены. Проверьте установку драйверов Steinberg.")
                p.terminate()
                return {"success": False, "error": "No ASIO devices found"}

    print(f"\nИспользуется устройство: {target_device['name']}")
    print(f"Доступно каналов: {target_device['channels']}")
    
    # Записываем 1 канал для теста
    test_channels = 1
    sample_rate = int(target_device['defaultSampleRate'])
    
    print(f"\nЗапись {duration} сек... (говорите в микрофон)")
    
    try:
        audio_data = record_audio(
            duration=duration,
            device_index=target_device['index'],
            channels=test_channels,
            sample_rate=sample_rate
        )
        
        # Анализ сигнала
        rms = np.sqrt(np.mean(audio_data**2))
        db_level = 20 * np.log10(rms + 1e-10)
        
        peak = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        threshold_db = config.MIC_CHECK_THRESHOLD_DB if hasattr(config, 'MIC_CHECK_THRESHOLD_DB') else -40
        
        print("\n" + "-"*60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("-"*60)
        print(f"Средний уровень (RMS): {db_level:.2f} дБ")
        print(f"Пиковый уровень: {peak_db:.2f} дБ")
        print(f"Порог обнаружения сигнала: {threshold_db} дБ")
        
        success = db_level > threshold_db
        
        if success:
            print("\n✅ МИКРОФОН РАБОТАЕТ!")
            print(f"   Сигнал обнаружен (уровень {db_level:.1f} дБ > порога {threshold_db} дБ)")
        else:
            print("\n❌ МИКРОФОН НЕ РАБОТАЕТ ИЛИ СИГНАЛ СЛИШКОМ ТИХИЙ")
            print(f"   Уровень сигнала {db_level:.1f} дБ < порога {threshold_db} дБ")
            print("\nРекомендации:")
            print("  1. Проверьте подключение микрофона к UR44C")
            print("  2. Включите фантомное питание +48V")
            print("  3. Увеличьте Gain на звуковой карте")
            
        p.terminate()
        return {
            "success": success,
            "rms_db": float(db_level),
            "peak_db": float(peak_db),
            "device_name": target_device['name']
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка при проверке: {e}")
        p.terminate()
        return {"success": False, "error": str(e)}
