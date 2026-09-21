# Speech-to-Text Pipeline

## Model choice

Transcription uses faster-whisper, the CTranslate2 build of Whisper. It is about
four times faster than the reference openai-whisper implementation at the same
word error rate. The default size is `small` with int8_float16 compute type,
which occupies roughly 1.2 GB of VRAM and reaches a real-time factor near 0.18
on an RTX 4060 laptop GPU.

The `medium` model lowers Ukrainian word error rate from 0.21 to 0.14 but needs
about 2.8 GB of VRAM and pushes the real-time factor to 0.42. Use it when
transcription accuracy matters more than latency and no reranker shares the card.

## Audio preprocessing

All uploads are converted to 16 kHz mono PCM before transcription. Stereo input
is downmixed by averaging channels rather than dropping one, because call
recordings frequently place each speaker on a separate channel. Compressed
formats such as webm, ogg and m4a are decoded with ffmpeg.

Voice activity detection is enabled by default. It removes long silences that
otherwise make Whisper emit repeated phrases, and it shortens processing time on
recordings with significant dead air.

## Known failure modes

Whisper repeats phrases when audio is noisy or nearly silent, so
`condition_on_previous_text` is disabled. Very quiet recordings below -45 dBFS
should be rejected rather than transcribed; the transcript is usually
hallucinated text unrelated to the audio.

Spoken queries reach the retriever through the query_rewrite prompt, which strips
filler words and restores domain terms. Passing raw transcripts to the retriever
cost 7 points of hit@4 on the voice subset of the evaluation set.
