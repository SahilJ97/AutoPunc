from sys import argv
import os
import json
import re

TRANSCRIPT, AUDIO_DIR, OUTPUT_SCP = argv[1], argv[2], argv[3]
GENTLE_PORT = "0.0.0.0:32768"  # may need to change. run 'docker ps' to find where Gentle image is listening


def format_scp_entry(ut_id, word_index, word, start, end, audio_fname):
    # <utterance_id>.<word_index>.<word> <command for getting audio segment>
    return f"{ut_id}.{word}.{word_index} ffmpeg -ss {start} -t {end} -i {audio_fname}\n"


if __name__ == "__main__":
    os.system(f"rm -f {OUTPUT_SCP} tmp.transcript tmp.align")
    with open(TRANSCRIPT, "r", encoding='utf-8-sig') as in_transcript:
        # matching is wrong! should get 000010.TXT, 000010001 with 000010001.WAV (in SPEAKER0001)
        for line in in_transcript:
            speaker = AUDIO_DIR.split("/")[-1].replace("SPEAKER", "")
            wav_basename = line.split("\t")[0]
            session = wav_basename.replace(f"0{speaker}", "")[0]
            audio_filename = f"{AUDIO_DIR}/SESSION{session}/{wav_basename}.WAV"
            utterance_text = next(in_transcript).strip()
            with open("tmp.transcript", "w+") as tmp:
                tmp.write(utterance_text)
            alignment_json = os.popen(
                f'curl -F "audio=@{audio_filename}" -F "transcript=@tmp.transcript" '
                f'"http://{GENTLE_PORT}/transcriptions?async=false"'
            ).read()
            alignments = json.loads(alignment_json)
            words = alignments["words"]
            for i in range(len(words)):
                word = words[i]
                print(word["word"])
                with open(OUTPUT_SCP, "a+") as out_scp:
                    out_scp.write(
                        format_scp_entry(wav_basename, i, word["word"], word["start"], word["end"], audio_filename)
                    )
            os.system(f"rm -f tmp.transcript tmp.align")

            # utterance id: SPEAKER ID followed by utterance  #
            # need to unzip WAVE directory contents!
            # common base: PART<PART>/DATA/CHANNEL0/
            # WAV: /WAVE/SPEAKER<SPEAKER_ID>/SESSION<SESSION #>/<SESSION #><utterance ID>.wav
            # TXT: /SCRIPT/<SPEAKER_ID>.TXT, and each TXT file contains utterance IDs. no session #s?

"""format:
{
  "transcript": "Mary and her family were moving to another city\n",
  "words": [
    {
      "alignedWord": "mary",
      "case": "success",
      "end": 0.8400000000000001,
      "endOffset": 4,
      "phones": [
        {
          "duration": 0.11,
          "phone": "m_B"
        },
        {
          "duration": 0.09,
          "phone": "eh_I"
        },
        {
          "duration": 0.09,
          "phone": "r_I"
        },
        {
          "duration": 0.09,
          "phone": "iy_E"
        }
      ],
      "start": 0.46,
      "startOffset": 0,
      "word": "Mary"
    },
    ...
"""
