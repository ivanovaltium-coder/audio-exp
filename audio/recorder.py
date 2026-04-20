# audio/recorder.py
import sounddevice as sd
import numpy as np
import config


def get_device_channels(device=None):
    """
    Определяет максимальное количество входных каналов устройства.
    
    Параметры:
        device: int или str — устройство ввода (по умолчанию None = системное по умолчанию).
    
    Возвращает:
        int — максимальное количество входных каналов.
    """
    if device is None:
        device = sd.default.device[0]
    
    if device is None:
        return 1
    
    try:
        dev_info = sd.query_devices(device)
        return max(1, dev_info['max_input_channels'])
    except Exception:
        return 1


def record_audio(duration=None, samplerate=None, device=None, channels=None, use_single_channel=None, active_channel=None):
    """
    Записывает аудио с микрофона.
    Поддерживает запись с 1 или 4 каналов. В первой версии используется 1 канал.

    Параметры:
        duration: float — длительность записи в секундах.
                  Если None, используется config.WINDOW_SEC.
        samplerate: int — частота дискретизации.
                    Если None, используется config.SAMPLE_RATE.
        device: int или str — устройство ввода (по умолчанию None = системное по умолчанию).
        channels: int — количество каналов для записи (1 или 4). 
                    Если None, берётся из config.NUM_CHANNELS.
        use_single_channel: bool — если True, используется только 1 канал из многоканальной записи.
                               Если None, берётся из config.USE_SINGLE_CHANNEL.
        active_channel: int — индекс активного канала (0-3), используется при use_single_channel=True.
                           Если None, берётся из config.ACTIVE_CHANNEL.

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
    
    # Определяем реальное количество каналов устройства
    max_channels = get_device_channels(device)
    
    # Если channels не указан, используем конфиг, но не больше чем поддерживает устройство
    if channels is None:
        channels = min(config.NUM_CHANNELS, max_channels)
    else:
        # Ограничиваем количеством каналов устройства
        channels = min(channels, max_channels)
    
    # Запись с микрофона
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
        device: int или str — устройство ввода (по умолчанию None = системное по умолчанию).
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
            try:
                dev_info = sd.query_devices(device)
                print(f"\nУстройство: {dev_info['name']}")
                print(f"Тип: {'Ввод' if dev_info['max_input_channels'] > 0 else 'Вывод'}")
                print(f"Каналы: {dev_info['max_input_channels']} входных / {dev_info['max_output_channels']} выходных")
                print(f"Частота: {dev_info['default_samplerate']} Гц")
            except Exception as e:
                print(f"Не удалось получить информацию об устройстве: {e}")
        else:
            default_device = sd.default.device[0]
            if default_device is not None:
                dev_info = sd.query_devices(default_device)
                print(f"\nИспользуется устройство по умолчанию: {dev_info['name']}")
            else:
                print("\nУстройство ввода по умолчанию не найдено!")
    
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
    """Выводит список доступных аудиоустройств."""
    print(sd.query_devices())
    print("\nУстройство ввода по умолчанию:", sd.default.device[0])


if __name__ == "__main__":
    # Тестовый запуск для проверки микрофона
    result = check_microphone(verbose=True)
    print(f"\nРезультат: {result['message']}")
