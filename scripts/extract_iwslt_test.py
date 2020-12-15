"""
Usage:
$ python3 extract_iwslt_test.py INPUT_XML AUDIO_DIR OUTPUT_DIR
"""

import re
import os
import shutil
from sys import argv
import json
import glob
import subprocess

GENTLE_PORT = "0.0.0.0:32768"  # May need to change. run 'docker ps' to find where Gentle image is listening
INPUT_XML, AUDIO_DIR, OUTPUT_DIR = argv[1], argv[2], argv[3]


def format_scp_entry(word_index, word, start, end, audio_fname):
    # <word>.<word_index> <command for getting audio segment>
    return f"{word}.{word_index} ffmpeg -loglevel panic -ss {start} -t {end} -i {audio_fname} -f wav - | \n"


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    with open(INPUT_XML, "r") as xml_file:
        xml = xml_file.read()
        speeches = re.findall(
            r"<talkid>(.+)</talkid>([\s\S]+?)</doc>",
            xml
        )
        for talk_id, speech_xml in speeches:

            # Write speech transcript
            if os.path.exists("tmp"):
                shutil.rmtree("tmp")
            os.mkdir("tmp")
            segments = re.findall(r"<seg id=\"[0-9]+\"> ([\S\s]+?)\s+?</seg>", speech_xml)
            speech = " ".join(segments)
            with open("tmp/transcript.txt", "w") as transcript_file:
                transcript_file.write(speech)

            # Convert audio to .wav
            sph_file = glob.glob(f"{AUDIO_DIR}/*talkid{talk_id}.sph")[0]
            wav_file = sph_file.replace(".sph", ".wav")
            subprocess.run(f"sph2pipe {sph_file} {wav_file}", shell=True)

            # Perform forced alignment
            alignment_json = os.popen(
                f'curl -F "audio=@{wav_file}" -F "transcript=@tmp/transcript.txt" '
                f'"http://{GENTLE_PORT}/transcriptions?async=false"'
            ).read()
            alignments = json.loads(alignment_json)
            words = alignments["words"]

            # Write SCP file
            with open(f"{OUTPUT_DIR}/{talk_id}-wav.scp", "w+") as out_scp:
                for i in range(len(words)):
                    word = words[i]
                    try:
                        out_scp.write(
                            format_scp_entry(
                                i, word["word"], word["start"], word["end"], wav_file
                            )
                        )
                    except KeyError:
                        print(f"Failed to find alignment for word {word['word']}")

            # Generate datafile from SCP
            os.system(
                f"bash scp_to_data.sh {OUTPUT_DIR}/{talk_id}-wav.scp tmp/transcript.txt {OUTPUT_DIR}/{talk_id}.data"
            )

            shutil.rmtree("tmp")
