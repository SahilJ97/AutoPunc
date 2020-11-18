import re
import wget
import requests
import os
import shutil
from sys import argv
import json

GENTLE_PORT = "0.0.0.0:32768"  # May need to change. run 'docker ps' to find where Gentle image is listening
INPUT_XML, OUTPUT_DIR = argv[1], argv[2]


def format_scp_entry(word_index, word, start, end, audio_fname):
    # <utterance_id>.<word_index>.<word> <command for getting audio segment>
    return f"{word}.{word_index} ffmpeg -loglevel panic -ss {start} -t {end} -i {audio_fname} -f wav - | \n"


if __name__ == "__main__":
    with open(INPUT_XML, "r") as xml_file:
        xml = xml_file.read()
        speeches = re.findall(
            r"<url>(.+)</url>([\s\S]+?)</doc>",
            xml
        )
        for speech_url, speech_xml in speeches:
            os.mkdir("tmp")
            html = requests.get(speech_url).text
            video_url = re.findall(r'"medium":"(.+?)"', html)
            wget.download(video_url, out="tmp/video.mp4")
            os.system(f"ffmpeg -i tmp/video.mp4 -ar 16000 {OUTPUT_DIR}/audio.wav")
            segments = re.findall(r"<seg id=\"[0-9]+\"> ([/S/s]+?)\s+</seg>", speech_xml)
            speech = " ".join(segments)
            with open("tmp/transcript.txt", "w") as transcript_file:
                transcript_file.write(speech)

            alignment_json = os.popen(
                f'curl -F "audio=@{OUTPUT_DIR}/audio.wav" -F "transcript=@tmp/transcript.txt" '
                f'"http://{GENTLE_PORT}/transcriptions?async=false"'
            ).read()
            alignments = json.loads(alignment_json)
            words = alignments["words"]
            for i in range(len(words)):
                word = words[i]
                print(word["word"])
                with open(f"{OUTPUT_DIR}/wav.scp", "a+") as out_scp:
                    try:
                        out_scp.write(
                            format_scp_entry(
                                i, word["word"], word["start"], word["end"], f"{OUTPUT_DIR}/audio.wav"
                            )
                        )
                    except KeyError:
                        print(f"ERROR: Failed to find alignment for word {word['word']}")
            shutil.rmtree("tmp")  # No! save audio file! other stuff can go in tmp.
