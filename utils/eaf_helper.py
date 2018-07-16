import xml.etree.ElementTree
import os

import pandas as pd

def eaf2df(eaf_file):
    e = xml.etree.ElementTree.parse(eaf_file).getroot()

    timeslots = {ts.attrib["TIME_SLOT_ID"]: int(ts.attrib["TIME_VALUE"]) for ts in
                 e.findall("TIME_ORDER")[0].findall("TIME_SLOT")}

    audio_file = os.path.basename(e.findall("HEADER")[0].findall("MEDIA_DESCRIPTOR")[0].attrib["MEDIA_URL"])

    transcriptions = [tier for tier in e.findall("TIER") if tier.attrib["LINGUISTIC_TYPE_REF"] == "transcription"]

    utterances = []
    for tr in transcriptions:
        for ann in tr.findall("ANNOTATION"):
            alignable_annotation = ann.findall("ALIGNABLE_ANNOTATION")[0]
            annotation_value = alignable_annotation.findall("ANNOTATION_VALUE")[0]
            utterances.append({
                "file_path": eaf_file,
                "file": os.path.basename(eaf_file).split(".")[0].split("-")[0],
                "participant": tr.attrib["PARTICIPANT"],
                "tier_id": tr.attrib["TIER_ID"],
                "annotation_id": alignable_annotation.attrib["ANNOTATION_ID"],
                "timeslot_start": alignable_annotation.attrib["TIME_SLOT_REF1"],
                "timeslot_end": alignable_annotation.attrib["TIME_SLOT_REF2"],
                "timeslot_start_ms": timeslots[alignable_annotation.attrib["TIME_SLOT_REF1"]],
                "timeslot_end_ms": timeslots[alignable_annotation.attrib["TIME_SLOT_REF2"]],
                "annotation": annotation_value.text,
                "audio_file": audio_file
            })

    return pd.DataFrame(utterances)