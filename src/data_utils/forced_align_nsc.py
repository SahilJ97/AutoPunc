from sys import argv
import os
import json

TRANSCRIPT, AUDIO_DIR, OUTPUT_SCP = argv[1], argv[2], argv[3]
GENTLE_PORT = "0.0.0.0:32768"  # May need to change. run 'docker ps' to find where Gentle image is listening
SEEN_SENTENCES = "seen.txt"


def format_scp_entry(ut_id, word_index, word, start, end, audio_fname):
    # <utterance_id>.<word_index>.<word> <command for getting audio segment>
    return f"{ut_id}.{word}.{word_index} ffmpeg -loglevel panic -ss {start} -t {end} -i {audio_fname} -f wav - | \n"


if __name__ == "__main__":
    seen_sentences = set()
    try:
        with open(SEEN_SENTENCES, "r") as seen_file:
            for sentence in seen_file:
                seen_sentences.add(sentence.strip())
    except FileNotFoundError:
        pass

    # Forced alignment, ignoring previously-seen sentences
    os.system(f"rm -f {OUTPUT_SCP} tmp.transcript tmp.align")
    speaker = AUDIO_DIR.split("/")[-1].replace("SPEAKER", "")
    with open(TRANSCRIPT, "r", encoding='utf-8-sig') as in_transcript:
        for line in in_transcript:
            _ = next(in_transcript)  # ignore next line
            flac_basename, utterance_text = line.split("\t")
            utterance_text = utterance_text.strip()

            # Duplicate control
            if utterance_text in seen_sentences:
                continue
            seen_sentences.add(utterance_text)
            with open(SEEN_SENTENCES, "a+") as seen_file:
                seen_file.write(utterance_text + "\n")

            session = flac_basename.replace(f"0{speaker}", "")[0]
            audio_filename = f"{AUDIO_DIR}/SESSION{session}/{flac_basename}.flac"
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
                    try:
                        out_scp.write(
                            format_scp_entry(
                                flac_basename, i, word["word"], word["start"], word["end"], audio_filename
                            )
                        )
                    except KeyError:
                        print(f"ERROR: Failed to find alignment for word {word['word']}")
            os.system(f"rm -f tmp.transcript tmp.align")
