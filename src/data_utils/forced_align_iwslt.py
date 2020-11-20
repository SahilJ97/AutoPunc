import re
import os
import shutil
from sys import argv
import json
import glob

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
            shutil.rmtree("tmp")
            os.mkdir("tmp")
            segments = re.findall(r"<seg id=\"[0-9]+\"> ([\S\s]+?)\s+?</seg>", speech_xml)
            speech = " ".join(segments)
            with open("tmp/transcript.txt", "w") as transcript_file:
                transcript_file.write(speech)

            sph_file = glob.glob(f"{AUDIO_DIR}/*talkid{talk_id}.sph")[0]
            wav_file = sph_file.replace(".sph", ".wav")
            os.system(f"sph2pipe {sph_file} {wav_file}")

            alignment_json = os.popen(
                f'curl -F "audio=@{wav_file}" -F "transcript=@tmp/transcript.txt" '
                f'"http://{GENTLE_PORT}/transcriptions?async=false"'
            ).read()
            alignments = json.loads(alignment_json)
            words = alignments["words"]
            for i in range(len(words)):
                word = words[i]
                with open(f"{OUTPUT_DIR}/wav.scp", "a+") as out_scp:
                    try:
                        out_scp.write(
                            format_scp_entry(
                                i, word["word"], word["start"], word["end"], f"{OUTPUT_DIR}/audio.wav"
                            )
                        )
                    except KeyError:
                        print(f"ERROR: Failed to find alignment for word {word['word']}")
            shutil.rmtree("tmp")
