#!/usr/bin/env python3
"""Generate native IDL/JSON conversion from the authoritative ROS 2 .msg files.

This intentionally accepts only this project's small fixed/bounded subset. It
does not pretend to replace rosidl_adapter for arbitrary ROS interfaces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("JointCmd", "JointState", "IMUState", "LowCmd", "LowState",
         "ControlRequest", "ControlReply")
BASIC = {"uint8": "octet", "uint64": "unsigned long long",
         "float64": "double", "bool": "boolean"}
FIELD = re.compile(r"(uint8|uint64|float64|bool|string<=\d+|[A-Z]\w*)(\[(?:<=)?\d+\])? ([a-z]\w*)$")


def schemas():
    result = {}
    for name in NAMES:
        fields = []
        for line in (ROOT / "ros2/opensprite_msgs/msg" / (name + ".msg")).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            match = FIELD.fullmatch(line)
            if not match:
                raise ValueError(f"unsupported authoritative .msg field: {name}: {line}")
            kind, array, field = match.groups()
            if kind not in BASIC and not kind.startswith("string<=") and kind not in result:
                raise ValueError(f"unknown or forward message reference: {kind}")
            fields.append((kind, array, field))
        result[name] = fields
    return result


def scalar_lines(kind, value, target):
    if kind.startswith("string<="):
        bound = int(kind.split("<=")[1])
        return [f'if (!{value}.is_string() || {value}.get_ref<const std::string&>().size() > {bound}) throw std::invalid_argument("invalid bounded UTF-8 string");',
                f'{target} = {value}.get<std::string>();']
    if kind in ("uint8", "uint64"):
        maximum = "255ULL" if kind == "uint8" else "UINT64_MAX"
        return [f'if (!({value}.is_number_unsigned() || ({value}.is_number_integer() && {value}.get<std::int64_t>() >= 0)) || {value}.get<std::uint64_t>() > {maximum}) throw std::invalid_argument("invalid unsigned integer");',
                f'{target} = {value}.get<std::{"uint8_t" if kind == "uint8" else "uint64_t"}>();']
    if kind == "float64":
        return [f'if (!{value}.is_number() || !std::isfinite({value}.get<double>())) throw std::invalid_argument("nonfinite or nonnumeric value");',
                f'{target} = {value}.get<double>();']
    if kind == "bool":
        return [f'if (!{value}.is_boolean()) throw std::invalid_argument("invalid boolean");',
                f'{target} = {value}.get<bool>();']
    return [f'from_json({value}, {target});']


def outputs():
    schema = schemas()
    digest = hashlib.sha256(b"".join((ROOT / "ros2/opensprite_msgs/msg" / (name + ".msg")).read_bytes() for name in NAMES)).hexdigest()
    idl = ["// Generated from ros2/opensprite_msgs/msg by tools/generate_dds_schema.py.",
           f"// Schema SHA256: {digest}", "module opensprite_msgs { module msg {"]
    cpp = ['// Generated from authoritative .msg; do not edit.', '#pragma once',
           '#include "OpenSprite.h"', '#include <nlohmann/json.hpp>', '#include <cmath>',
           '#include <cstdint>', '#include <stdexcept>', 'namespace opensprite_msgs { namespace msg {']
    validation = ['// Generated from authoritative .msg; do not edit.', '#pragma once',
        '#include "OpenSprite.h"', '#include <cmath>', '#include <stdexcept>',
        '#include <string>', 'namespace opensprite_msgs { namespace msg {',
        'template<class String> inline void validate_wire_string(const String& value, size_t bound) {',
        '  if (value.size()>bound) throw std::invalid_argument("invalid bounded string");',
        '  const char* bytes=value.c_str();',
        '  for (size_t i=0; i<value.size();) {',
        '    unsigned c=static_cast<unsigned char>(bytes[i++]), n=0, minimum=0;',
        '    if (c==0) throw std::invalid_argument("embedded NUL in string");',
        '    if (c<128) continue;',
        '    if ((c&0xe0)==0xc0) {c&=0x1f;n=1;minimum=0x80;}',
        '    else if ((c&0xf0)==0xe0) {c&=0x0f;n=2;minimum=0x800;}',
        '    else if ((c&0xf8)==0xf0) {c&=0x07;n=3;minimum=0x10000;}',
        '    else throw std::invalid_argument("invalid UTF-8 string");',
        '    if (i+n>value.size()) throw std::invalid_argument("truncated UTF-8 string");',
        '    while (n--) {unsigned b=static_cast<unsigned char>(bytes[i++]); if ((b&0xc0)!=0x80) throw std::invalid_argument("invalid UTF-8 continuation"); c=(c<<6)|(b&0x3f);}',
        '    if (c<minimum || c>0x10ffff || (c>=0xd800 && c<=0xdfff)) throw std::invalid_argument("invalid UTF-8 scalar");',
        '  }', '}',
    ]

    def validate_scalar(kind, value):
        if kind.startswith("string<="):
            return f"validate_wire_string({value}, {kind.split('<=')[1]});"
        if kind == "float64":
            return f'if (!std::isfinite({value})) throw std::invalid_argument("nonfinite wire value");'
        if kind not in BASIC:
            return f"validate_wire({value});"
        return ""
    for name, fields in schema.items():
        validation.append(f'inline void validate_wire(const {name}& v) {{')
        for kind, array, field in fields:
            value = f"v.{field}()"
            if array:
                if array.startswith("[<="):
                    validation.append(f'  if ({value}.size()>{array[3:-1]}) throw std::invalid_argument("wire sequence exceeds bound");')
                statement = validate_scalar(kind, "item")
                if statement:
                    validation.append(f"  for (const auto& item : {value}) {{ {statement} }}")
            else:
                statement = validate_scalar(kind, value)
                if statement:
                    validation.append("  "+statement)
        validation.append("}")
        idl.append(f"struct {name} {{")
        for kind, array, field in fields:
            wire = BASIC.get(kind, kind)
            if kind.startswith("string<="):
                wire = f"string<{kind.split('<=')[1]}>"
            if array and array.startswith("[<="):
                idl.append(f"  sequence<{wire}, {array[3:-1]}> {field};")
            else:
                idl.append(f"  {wire} {field}{array or ''};")
        idl.append("};")
        cpp.append(f'inline void from_json(const nlohmann::json& j, {name}& v) {{')
        cpp.append(f'  if (!j.is_object() || j.size() != {len(fields)}) throw std::invalid_argument("invalid {name} fields");')
        for kind, array, field in fields:
            value, target = f'j.at("{field}")', f'v.{field}()'
            cpp.append("  {")
            if array:
                bound = int(array[3:-1] if array.startswith("[<=") else array[1:-1])
                op = ">" if array.startswith("[<=") else "!="
                cpp.append(f'    if (!{value}.is_array() || {value}.size() {op} {bound}) throw std::invalid_argument("invalid {field} length");')
                if array.startswith("[<="):
                    cpp.append(f'    {target}.resize({value}.size());')
                cpp.append(f'    for (size_t i = 0; i < {value}.size(); ++i) {{')
                cpp.extend("      " + x for x in scalar_lines(kind, value + "[i]", target + "[i]"))
                cpp.append("    }")
            else:
                cpp.extend("    " + x for x in scalar_lines(kind, value, target))
            cpp.append("  }")
        cpp.append("}")
        cpp.append(f'inline void to_json(nlohmann::json& j, const {name}& v) {{')
        cpp.append("  j = nlohmann::json::object();")
        cpp.extend(f'  j["{field}"] = v.{field}();' for _, _, field in fields)
        cpp.append("}")
    idl.append("}; };")
    cpp.append("}} // namespace opensprite_msgs::msg")
    validation.append("}} // namespace opensprite_msgs::msg")
    return {ROOT / "native-dds/idl/OpenSprite.idl": "\n".join(idl) + "\n",
            ROOT / "native-dds/src/wire_json.hpp": "\n".join(cpp) + "\n",
            ROOT / "native-dds/src/wire_validation.hpp": "\n".join(validation) + "\n"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--record-native", action="store_true",
                        help="record the pinned Fast DDS Gen output after regeneration")
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise SystemExit(f"generated file out of date: {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    native = ROOT / "native-dds/generated"
    manifest_path = native / "manifest.json"
    if args.check or args.record_native:
        manifest = {"generator": "Fast DDS Gen 2.3.0 -typeros2",
                    "idl_sha256": hashlib.sha256((ROOT / "native-dds/idl/OpenSprite.idl").read_bytes()).hexdigest(),
                    "files": {name: hashlib.sha256((native / name).read_bytes()).hexdigest()
                              for name in ("OpenSprite.h", "OpenSprite.cxx", "OpenSpritePubSubTypes.h", "OpenSpritePubSubTypes.cxx")}}
        if args.check:
            if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
                raise SystemExit("native DDS types are stale or modified; run bash tools/regenerate_dds.sh")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
