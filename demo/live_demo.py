from collections.abc import Iterator
from typing import Any

import sounddevice
import voice2gender


def audio_stream_pc(
    *,
    samplerate: int = 16_000,
    channels: int = 1,
    blocksize: int = 8_000,
    device: int | str | None = None,
    latency: str = "low",
) -> Iterator[bytes]:
    """以生成器形式持续读取麦克风并产出原始 PCM 音频块。

    读取采用 ``sounddevice.RawInputStream``，避免依赖 NumPy，并固定使用
    ``int16`` 采样格式。默认参数对应 FunASR 常用的 16 kHz、单声道音频，
    每个生成块约 0.5 秒（8,000 帧）。调用方可通过 ``generator.close()``
    停止读取，函数会在生成器退出时停止并关闭音频流。

    Args:
        samplerate: 采样率，单位为 Hz，必须为正整数。
        channels: 输入声道数，必须为正整数。FunASR 通常使用单声道输入。
        blocksize: 每次读取的帧数，必须为正整数。
        device: 输入设备索引或名称；``None`` 使用 sounddevice 的默认设备。
        latency: sounddevice 延迟策略，通常为 ``"low"`` 或 ``"high"``。

    Yields:
        一个个非空的 ``bytes`` 对象，每个对象包含交错排列的 int16 PCM 数据。

    Raises:
        ValueError: 采样率、声道数或块大小不是正整数时抛出。
        sounddevice.PortAudioError: 无法打开或读取指定输入设备时由 sounddevice
            原样抛出。
    """
    _validate_positive_integer("samplerate", samplerate)
    _validate_positive_integer("channels", channels)
    _validate_positive_integer("blocksize", blocksize)

    stream: Any = sounddevice.RawInputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        device=device,
        channels=channels,
        dtype="int16",
        latency=latency,
    )
    try:
        stream.start()
        while True:
            data, _overflowed = stream.read(blocksize)
            pcm = bytes(data)
            if pcm:
                yield pcm
    finally:
        # 显式清理覆盖启动失败、读取异常和 generator.close() 等退出路径。
        stream.stop()
        stream.close()


def _validate_positive_integer(name: str, value: int) -> None:
    """校验音频参数为正整数，避免底层库产生难定位的错误。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须为正整数，实际值为 {value!r}")


if __name__ == "__main__":
    max_len = 10
    seq = []
    for pcm in audio_stream_pc():
        seq.append(pcm)
        print(voice2gender.predict(seq))
