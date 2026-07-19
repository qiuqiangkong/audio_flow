from xml.etree.ElementTree import Element, tostring
import xml.etree.ElementTree as ET
import re

def check_xml(xml: str) -> bool:
    try:
        ET.fromstring(xml)
    except:
        raise ValueError(f"Incorrect XML format: {xml}")


def xml_to_text(xml: str, sep="\n") -> str:
    if xml == "":
        return ""

    def walk(node):
        lines = [node.tag]
        lines += [f"{k}: {v}" for k, v in node.attrib.items()]
        for child in node:
            lines += walk(child)
        return lines

    try:
        return sep.join(walk(ET.fromstring(xml)))
    except:
        from IPython import embed; embed(using=False); os._exit(0)


def batch_xml_to_text(xmls: list[str]) -> list[str]:
    return [xml_to_text(xml) for xml in xmls]


def get_xml_attr(xml: str, attr: str, sep="\n") -> str:
    if xml == "":
        return ""

    def walk(node):
        lines = [node.attrib.get(attr, "")]
        for child in node:
            lines += walk(child)
        return lines

    return sep.join(walk(ET.fromstring(xml)))


def batch_get_xml_attr(xmls: list[str], attr: str) -> list[str]:
    return [get_xml_attr(xml, attr) for xml in xmls]


'''
def dict_to_xml(data_dict: dict) -> list[str]:
    """Convert dict to XML. E.g.:

    Input:
        data_dict = {
            "speech": {
                "content": "hello world!",
                "language": "en",
                "emotion": "angry",
            },
            "music": {
                "caption": "a boy is playing guitar and piano.",
                "genre": "jazz",
                "bpm": "80",
            },
        }

    Output:
        <speech language="en" emotion="angry">hello world!</speech>
        <music genre="jazz" bpm="80">a boy is playing guitar and piano.</music>
    """

    xml = []

    for key, data in data_dict.items():

        field = next((k for k in ["content", "caption"] if k in data), "")
        text = data.get(field, "")

        attrs = {
            k: str(v)
            for k, v in data.items()
            if k != field and v not in ["", None]
        }

        root = Element(key, attrs)
        root.text = text

        xml.append(tostring(root, encoding="unicode"))
    
    return xml


def xml_to_str(xml):
    return "".join(xml)


def str_to_xml(s):
    xml = re.findall(r"<\w+[^>]*>.*?</\w+>", s)
    return xml
'''