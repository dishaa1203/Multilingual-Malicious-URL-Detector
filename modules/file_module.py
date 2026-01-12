import pefile

def analyze_file(filepath: str):
    try:
        pe = pefile.PE(filepath)
        suspicious = any(
            b"UPX" in section.Name or b".text" not in section.Name
            for section in pe.sections
        )
        confidence = 0.9 if suspicious else 0.1
        return confidence, ["Suspicious PE structure" if suspicious else "Clean"]
    except:
        return 0.0, ["Not a PE file"]
