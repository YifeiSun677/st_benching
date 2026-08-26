"""
Split construction. Kept separate so the within-patient (LOSO) and
leave-one-patient-out (LOPO) runs share one definition and one code path.
"""
import glob
import os


def all_sections(root):
    hits = []
    for ext in ("tsv.gz", "tsv"):
        hits += glob.glob(os.path.join(root, "ST-cnts", f"*.{ext}"))
    return sorted({os.path.basename(p).split(".")[0] for p in hits})


def all_patients(root):
    return sorted({s[0] for s in all_sections(root)})


def sections_of(root, patient):
    return sorted(s for s in all_sections(root) if s[0] == patient)


def lopo_split(root, held_out_patient):
    """Train on every section of every other patient; test on all of theirs."""
    sections = all_sections(root)
    test = [s for s in sections if s[0] == held_out_patient]
    train = [s for s in sections if s[0] != held_out_patient]
    if not test:
        raise SystemExit(f"no sections for patient {held_out_patient}")
    return train, test


def loso_split(root, patient, held_out_section):
    """Within-patient: train on that patient's other sections."""
    sections = sections_of(root, patient)
    if held_out_section not in sections:
        raise SystemExit(f"{held_out_section} not in {sections}")
    return [s for s in sections if s != held_out_section], [held_out_section]


def parse_sections(arg):
    return [s.strip() for s in arg.split(",") if s.strip()]
