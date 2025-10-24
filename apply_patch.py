import pathlib
from textwrap import dedent
path = pathlib.Path('service/whisper_worker.py')
text = path.read_text()
old = dedent('''            if wasapi_settings_cls and sys_index is not None:
                max_output = 2
                max_input = 0
                if devices and sys_index < len(devices):
                    try:
                        max_output = int(devices[sys_index].get("max_output_channels") or max_output)
                    except Exception:
                        max_output = 2
                    try:
                        max_input = int(devices[sys_index].get("max_input_channels") or 0)
                    except Exception:
                        max_input = 0
                candidate_channels = []
                if max_output > 0:
                    candidate_channels.append(min(2, max_output))
                    if max_output >= 1 and 1 not in candidate_channels:
                        candidate_channels.append(1)
                else:
                    candidate_channels = [2, 1]
                if max_input > 0:
                    filtered = []
                    for ch in candidate_channels:
                        adjusted = max(1, min(ch, max_input))
                        if adjusted not in filtered:
                            filtered.append(adjusted)
                    candidate_channels = filtered or [1]
                else:
                    if 1 not in candidate_channels:
                        candidate_channels.append(1)
                attempt_errors = []
                for ch in candidate_channels:
                    settings = wasapi_settings_cls(exclusive=False, auto_convert=True)
                    try:
                        settings.loopback = True
                    except AttributeError:
                        pass
                    sys_kwargs = {
                        "device": sys_index,
                        "samplerate": SAMPLE_RATE,
                        "blocksize": BLOCK_FRAMES,
                        "dtype": "float32",
                        "channels": ch,
                        "callback": _make_audio_callback("system"),
                        "extra_settings": settings,
                    }
                    try:
                        system_stream = sd.InputStream(**sys_kwargs)
                        system_stream.start()
                        _streams.append(system_stream)
                        system_started = True
                        try:
                            dev_name = devices[sys_index]["name"] if devices and sys_index < len(devices) else str(sys_index)
                        except Exception:
                            dev_name = str(sys_index)
                        print(f"[whisper_worker] system loopback stream started (WASAPI) -> {dev_name} (channels={ch})")
                        break
                    except Exception as e:
                        attempt_errors.append((ch, str(e)))
                        print(f"[whisper_worker] failed to start WASAPI loopback with channels={ch}: {e}")
                if not system_started:
                    if attempt_errors:
                        print(f"[whisper_worker] WASAPI loopback attempts exhausted: {attempt_errors}")
                    else:
                        print("[whisper_worker] WASAPI loopback attempts exhausted")
''')
new = dedent('''            if wasapi_settings_cls and sys_index is not None:
                max_output = 2
                max_input = 0
                if devices and sys_index < len(devices):
                    try:
                        max_output = int(devices[sys_index].get("max_output_channels") or max_output)
                    except Exception:
                        max_output = 2
                    try:
                        max_input = int(devices[sys_index].get("max_input_channels") or 0)
                    except Exception:
                        max_input = 0
                raw_candidates = []
                if max_output > 0:
                    raw_candidates.extend([max_output, 2, 1])
                else:
                    raw_candidates.extend([2, 1])
                if max_input > 0:
                    raw_candidates.append(max_input)
                candidate_channels = []
                for ch in raw_candidates:
                    try:
                        ch = int(ch)
                    except Exception:
                        continue
                    if ch <= 0:
                        continue
                    ch = max(1, ch)
                    if max_input > 0:
                        ch = min(ch, max_input)
                    if ch not in candidate_channels:
                        candidate_channels.append(ch)
                if not candidate_channels:
                    candidate_channels = [2, 1]
                attempt_errors = []
                for ch in candidate_channels:
                    settings = wasapi_settings_cls(exclusive=False, auto_convert=True)
                    try:
                        settings.loopback = True
                    except AttributeError:
                        pass
                    sys_kwargs = {
                        "device": sys_index,
                        "samplerate": SAMPLE_RATE,
                        "blocksize": BLOCK_FRAMES,
                        "dtype": "float32",
                        "channels": ch,
                        "callback": _make_audio_callback("system"),
                        "extra_settings": settings,
                    }
                    try:
                        system_stream = sd.InputStream(**sys_kwargs)
                        system_stream.start()
                        _streams.append(system_stream)
                        system_started = True
                        try:
                            dev_name = devices[sys_index]["name"] if devices and sys_index < len(devices) else str(sys_index)
                        except Exception:
                            dev_name = str(sys_index)
                        print(f"[whisper_worker] system loopback stream started (WASAPI) -> {dev_name} (channels={ch})")
                        break
                    except Exception as e:
                        attempt_errors.append((ch, str(e)))
                        print(f"[whisper_worker] failed to start WASAPI loopback with channels={ch}: {e}")
                if not system_started:
                    if attempt_errors:
                        print(f"[whisper_worker] WASAPI loopback attempts exhausted: {attempt_errors}")
                    else:
                        print("[whisper_worker] WASAPI loopback attempts exhausted")
''')
if old not in text:
    raise SystemExit('target block not found')
path.write_text(text.replace(old, new, 1))
