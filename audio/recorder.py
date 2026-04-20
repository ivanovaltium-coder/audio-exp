# audio/recorder.py
"""
Модуль записи аудио с поддержкой ASIO драйвера Steinberg UR44C.
Обеспечивает низкоуровневую запись с минимальной задержкой для точного фазового анализа.
"""
import sounddevice as sd
import numpy as np
import config
import time


def find_asio_device(device_name=None):
    """
    Поиск устройства ASIO по имени или возврат первого доступного ASIO устройства.
    
    Параметры:
        device_name: str — имя устройства для поиска (частичное совпадение).
    
    Возвращает:
        int или None — индекс устройства ASIO, или None если не найдено.
    """
    asio_devices = []
    
    # Проходим по всем хост-API
    for host_api_idx in range(sd.query_hostapis().shape[0]):
        host_api = sd.query_hostapis(host_api_idx)
        if 'ASIO' in host_api['name'].upper():
            # Нашли ASIO API, теперь ищем устройства в нём
            for dev_idx in host_api['devices']:
                dev_info = sd.query_devices(dev_idx)
                if dev_info['max_input_channels'] > 0:
                    asio_devices.append((dev_idx, dev_info['name']))
                    
                    # Если указано имя для поиска, проверяем совпадение
                    if device_name and device_name.lower() in dev_info['name'].lower():
                        print(f"✓ Найдено ASIO устройство: {dev_info['name']} (ID: {dev_idx})")
                        return dev_idx
    
    if asio_devices:
        # Возвращаем первое ASIO устройство если точное совпадение не найдено
        print(f"⚠ Устройство '{device_name}' не найдено. Используем первое доступное ASIO: {asio_devices[0][1]}")
        return asio_devices[0][0]
    
    return None


def get_asio_device_info(device=None):
    """
    Получение информации об ASIO устройстве.
    
    Параметры:
        device: int или str — устройство ввода.
    
    Возвращает:
        dict с информацией об устройстве или None.
    """
    try:
        if device is None:
            device = sd.default.device[0]
        
        dev_info = sd.query_devices(device)
        return {
            'id': device if isinstance(device, int) else None,
            'name': dev_info['name'],
            'input_channels': dev_info['max_input_channels'],
            'output_channels': dev_info['max_output_channels'],
            'samplerate': dev_info['default_samplerate']
        }
    except Exception:
        return None


def get_device_channels(device=None):
    """
    Определяет максимальное количество входных каналов устройства.
    
    Параметры:
        device: int или str — устройство ввода (по умолчанию None = системное по умолчанию).
    
    Возвращает:
        int — максимальное количество входных каналов.
    """
    if device is None:
        if config.USE_ASIO:
            # Пытаемся найти ASIO устройство
            device = find_asio_device(config.ASIO_DEVICE_NAME)
            if device is None:
                device = sd.default.device[0]
        else:
            device = sd.default.device[0]
    
    if device is None:
        return 1
    
    try:
        dev_info = sd.query_devices(device)
        return max(1, dev_info['max_input_channels'])
    except Exception:
        return 1


def record_audio(duration=None, samplerate=None, device=None, channels=None, 
                 use_single_channel=None, active_channel=None, use_callback=False):
    """
    Записывает аудио с микрофона с поддержкой ASIO драйвера.
    Поддерживает запись с 1 до 8 каналов. В первой версии используется 1 канал.
    
    Параметры:
        duration: float — длительность записи в секундах.
                  Если None, используется config.WINDOW_SEC.
        samplerate: int — частота дискретизации.
                    Если None, используется config.SAMPLE_RATE.
        device: int или str — устройство ввода. 
                Если None и USE_ASIO=True, автоматически ищется ASIO устройство.
        channels: int — количество каналов для записи (1-8). 
                    Если None, берётся из config.NUM_CHANNELS.
        use_single_channel: bool — если True, используется только 1 канал из многоканальной записи.
                               Если None, берётся из config.USE_SINGLE_CHANNEL.
        active_channel: int — индекс активного канала (0-7), используется при use_single_channel=True.
                           Если None, берётся из config.ACTIVE_CHANNEL.
        use_callback: bool — использовать callback режим для минимальной задержки.
    
    Возвращает:
        np.ndarray форма (N,) — аудиоданные с плавающей точкой в диапазоне [-1, 1] (моно).
    """
    if duration is None:
        duration = config.WINDOW_SEC
    if samplerate is None:
        samplerate = config.SAMPLE_RATE
    if use_single_channel is None:
        use_single_channel = config.USE_SINGLE_CHANNEL
    if active_channel is None:
        active_channel = config.ACTIVE_CHANNEL
    
    # Автоматический поиск ASIO устройства если не указано
    if device is None and config.USE_ASIO:
        device = find_asio_device(config.ASIO_DEVICE_NAME)
        if device is None:
            print("⚠ ASIO устройство не найдено, используем устройство по умолчанию")
            device = sd.default.device[0]
    
    # Определяем реальное количество каналов устройства
    max_channels = get_device_channels(device)
    
    # Если channels не указан, используем конфиг, но не больше чем поддерживает устройство
    if channels is None:
        channels = min(config.NUM_CHANNELS, max_channels)
    else:
        # Ограничиваем количеством каналов устройства
        channels = min(channels, max_channels)
    
    # Проверка диапазона активного канала
    if active_channel >= channels:
        print(f"⚠ Активный канал {active_channel} вне диапазона. Используется канал 0.")
        active_channel = 0
    
    print(f"🎙️ Запись: {duration} сек, {samplerate} Гц, {channels} кан., устройство ID: {device}")
    
    if use_callback:
        # Callback режим для минимальной задержки (рекомендуется для ASIO)
        frames = []
        buffer_size = getattr(config, 'ASIO_BUFFER_SIZE', 1024)
        
        def callback(indata, frame_count, time_info, status_flags):
            if status_flags:
                print(f"⚠ Статус: {status_flags}")
            frames.append(indata.copy())
        
        with sd.InputStream(samplerate=samplerate,
                           device=device,
                           channels=channels,
                           blocksize=buffer_size,
                           dtype='float32',
                           callback=callback):
            sd.sleep(int(duration * 1000))
        
        recording = np.concatenate(frames, axis=0)
    else:
        # Стандартный режим записи
        recording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype='float32',
            device=device
        )
        sd.wait()  # ждём окончания записи
    
    # Если записано несколько каналов, но нужен только один
    if use_single_channel and channels > 1:
        # recording имеет форму (samples, channels)
        if recording.ndim == 2:
            recording = recording[:, active_channel]
        else:
            # Если вдруг запись моно, оставляем как есть
            pass
    
    return recording.flatten()


def check_microphone(device=None, duration=2.0, samplerate=None, verbose=True):
    """
    Проверка работы конденсаторного микрофона через звуковую карту.
    Записывает короткий фрагмент и анализирует уровень сигнала.
    
    Параметры:
        device: int — ID устройства ввода (PyAudio device index).
        duration: float — длительность тестовой записи в секундах.
        samplerate: int — частота дискретизации. Если None, используется config.SAMPLE_RATE.
        verbose: bool — если True, выводит подробную информацию.
    
    Возвращает:
        dict с результатами проверки:
            - 'working': bool — работает ли микрофон (сигнал выше порога)
            - 'rms_db': float — среднеквадратичный уровень сигнала в дБ
            - 'peak_db': float — пиковый уровень сигнала в дБ
            - 'message': str — текстовое сообщение о результате
    """
    if samplerate is None:
        samplerate = config.SAMPLE_RATE
    
    if verbose:
        print("=" * 60)
        print("ПРОВЕРКА МИКРОФОНА")
        print("=" * 60)
        
        # Информация об устройстве
        if device is not None:
            p = pyaudio.PyAudio()
            try:
                dev_info = p.get_device_info_by_index(device)
                print(f"\nУстройство: {dev_info['name']}")
                print(f"Тип: {'Ввод' if dev_info['maxInputChannels'] > 0 else 'Вывод'}")
                print(f"Каналы: {dev_info['maxInputChannels']} входных / {dev_info['maxOutputChannels']} выходных")
                print(f"Частота: {dev_info['defaultSampleRate']} Гц")
            except Exception as e:
                print(f"Не удалось получить информацию об устройстве: {e}\n")
                print("Попробуйте запустить: python main.py --list-asio")
                print("для просмотра доступных ASIO устройств.")
                p.terminate()
                return {
                    'working': False,
                    'rms_db': -100.0,
                    'peak_db': -100.0,
                    'message': f'Ошибка устройства: {e}',
                    'audio_data': np.array([])
                }
            p.terminate()
        else:
            if config.USE_ASIO:
                # Пытаемся найти ASIO устройство
                device = find_asio_device(config.ASIO_DEVICE_NAME)
                if device is not None:
                    p = pyaudio.PyAudio()
                    dev_info = p.get_device_info_by_index(device)
                    print(f"\nНайдено ASIO устройство: {dev_info['name']} (ID: {device})")
                    p.terminate()
                else:
                    print("\n⚠ ASIO устройство не найдено.")
                    print("Попробуйте запустить: python main.py --list-asio")
                    print("и указать устройство явно: python main.py --check-mic --device <ID>")
                    return {
                        'working': False,
                        'rms_db': -100.0,
                        'peak_db': -100.0,
                        'message': 'ASIO устройство не найдено',
                        'audio_data': np.array([])
                    }
            else:
                p = pyaudio.PyAudio()
                try:
                    default_dev = p.get_default_input_device_info()
                    print(f"\nИспользуется устройство по умолчанию: {default_dev['name']}")
                except Exception:
                    print("\n⚠ Устройство по умолчанию не найдено или недоступно.")
                    print("Попробуйте запустить: python main.py --list-asio")
                    print("и указать устройство явно: python main.py --check-mic --device <ID>")
                    p.terminate()
                    return {
                        'working': False,
                        'rms_db': -100.0,
                        'peak_db': -100.0,
                        'message': 'Устройство по умолчанию не найдено',
                        'audio_data': np.array([])
                    }
                p.terminate()
    
    # Запись тестового фрагмента (всегда 1 канал для проверки микрофона)
    if verbose:
        print(f"\nЗапись {duration} сек... (говорите в микрофон)")
    
    recording = record_audio(
        duration=duration,
        samplerate=samplerate,
        device=device,
        channels=1,  # Для проверки используем 1 канал
        use_single_channel=True,
        active_channel=0
    )
    
    if len(recording) == 0:
        return {
            'working': False,
            'rms_db': -100.0,
            'peak_db': -100.0,
            'message': 'Не удалось записать аудио',
            'audio_data': np.array([])
        }
    
    # Анализ уровня сигнала
    # Вычисляем RMS (среднеквадратичное значение)
    rms = np.sqrt(np.mean(recording ** 2))
    
    # Преобразуем в дБ (относительно полной шкалы 0 dBFS)
    # Добавляем маленькое число для избежания log(0)
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Пиковое значение
    peak = np.max(np.abs(recording))
    peak_db = 20 * np.log10(peak + 1e-10)
    
    # Проверка: сигнал выше порога?
    threshold = config.MIC_CHECK_THRESHOLD_DB
    working = rms_db > threshold
    
    if verbose:
        print("\n" + "-" * 60)
        print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("-" * 60)
        print(f"Средний уровень (RMS): {rms_db:.2f} дБ")
        print(f"Пиковый уровень: {peak_db:.2f} дБ")
        print(f"Порог обнаружения сигнала: {threshold:.1f} дБ")
        print()
        
        if working:
            print("✓ МИКРОФОН РАБОТАЕТ!")
            print(f"  Сигнал обнаружен (уровень {rms_db:.1f} дБ > порога {threshold:.1f} дБ)")
            print("  Запись идёт с конденсаторного микрофона через звуковую карту.")
        else:
            print("✗ МИКРОФОН НЕ РАБОТАЕТ ИЛИ СИГНАЛ СЛИШКОМ ТИХИЙ")
            print(f"  Уровень сигнала {rms_db:.1f} дБ < порога {threshold:.1f} дБ")
            print("\nРекомендации:")
            print("  1. Убедитесь, что микрофон подключён к звуковой карте Steinberg UR44C")
            print("  2. Проверьте, что на микрофон подаётся фантомное питание +48V")
            print("  3. Увеличьте громкость входа на звуковой карте")
            print("  4. Говорите громче в микрофон во время теста")
            print("  5. Проверьте настройки устройства ввода в системе")
        
        print("=" * 60)
    
    return {
        'working': working,
        'rms_db': float(rms_db),
        'peak_db': float(peak_db),
        'message': 'Микрофон работает' if working else 'Микрофон не работает или сигнал слишком тихий',
        'audio_data': recording
    }


def list_devices():
    """Выводит список доступных аудиоустройств с информацией о Host API через PyAudio."""
    print("=" * 70)
    print("ДОСТУПНЫЕ HOST API:")
    print("=" * 70)
    
    p = pyaudio.PyAudio()
    
    # Выводим информацию о хост-API
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        print(f"\nID {i}: {api_info['name']}")
        print(f"   Устройств: {api_info['deviceCount']}")
        
        # Показываем устройства в этом API
        for j in range(api_info['deviceCount']):
            dev_idx = api_info['devices'][j]
            dev = p.get_device_info_by_index(dev_idx)
            print(f"   └─ ID {dev_idx}: {dev['name']}")
            print(f"      Входов: {dev['maxInputChannels']}, Выходов: {dev['maxOutputChannels']}")
            if 'ASIO' in api_info['name'].upper() and dev['maxInputChannels'] >= 4:
                print(f"      🎯 ПОДХОДИТ ДЛЯ ПЕЛЕНГАЦИИ (≥4 каналов)")
    
    # Устройства по умолчанию
    print("\n" + "=" * 70)
    print("УСТРОЙСТВА ПО УМОЛЧАНИЮ:")
    print("=" * 70)
    
    try:
        default_input = p.get_default_input_device_info()
        print(f"Ввод:  ID {default_input['index']} - {default_input['name']}")
    except Exception:
        print("Ввод:  не настроено")
    
    try:
        default_output = p.get_default_output_device_info()
        print(f"Вывод: ID {default_output['index']} - {default_output['name']}")
    except Exception:
        print("Вывод: не настроено")
    
    p.terminate()


def list_asio_devices():
    """Выводит только ASIO устройства для записи с UR44C через PyAudio."""
    print("=" * 70)
    print("ASIO УСТРОЙСТВА (рекомендуется для Steinberg UR44C):")
    print("=" * 70)
    
    p = pyaudio.PyAudio()
    found = False
    
    # Ищем ASIO API
    asio_api_idx = -1
    for i in range(p.get_host_api_count()):
        api_info = p.get_host_api_info_by_index(i)
        if "ASIO" in api_info['name'].upper():
            asio_api_idx = i
            print(f"\nHost API: {api_info['name']}")
            print("-" * 50)
            
            # Показываем все устройства в ASIO API
            for j in range(api_info['deviceCount']):
                dev_idx = api_info['devices'][j]
                dev_info = p.get_device_info_by_index(dev_idx)
                
                if dev_info['maxInputChannels'] > 0:
                    found = True
                    print(f"ID {dev_idx}: {dev_info['name']}")
                    print(f"   Входных каналов: {dev_info['maxInputChannels']}")
                    print(f"   Частота: {dev_info['defaultSampleRate']} Гц")
                    
                    if dev_info['maxInputChannels'] >= 8:
                        print(f"   🎯 ПОЛНАЯ ПОДДЕРЖКА UR44C (8 каналов)")
                    elif dev_info['maxInputChannels'] >= 4:
                        print(f"   ✓ ПОДХОДИТ ДЛЯ ПЕЛЕНГАЦИИ (4+ канала)")
    
    if not found:
        print("\n⚠ ASIO устройства не найдены!")
        print("Проверьте установку драйверов Steinberg UR44C.")
        print("Убедитесь, что устройство не занято другой программой (DAW, VoiceMeeter).")
    
    p.terminate()
    print("=" * 70)


if __name__ == "__main__":
    # Тестовый запуск для проверки микрофона
    result = check_microphone(verbose=True)
    print(f"\nРезультат: {result['message']}")
