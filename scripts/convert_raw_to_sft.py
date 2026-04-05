import json
from pathlib import Path
import re


def format_input(obs):
    title = obs.get("title", "")
    url = obs.get("url", "")
    buttons = ", ".join(obs.get("visible_buttons", []))
    search_box = obs.get("search_box_present", False)

    inputs = []
    for x in obs.get("visible_inputs", []):
        input_type = x.get("type", "")
        name = x.get("name", "")
        placeholder = x.get("placeholder", "")
        value = x.get("value", "")

        if input_type == "hidden":
            continue

        # keep only useful fields
        if (
            name in ["q", "email", "username", "password"]
            or "search" in str(placeholder).lower()
            or input_type in ["email", "password"]
        ):
            inputs.append(
                f'{input_type} input name={name} placeholder={placeholder} value={value}'
            )

    return (
        f"Title: {title}\n"
        f"URL: {url}\n"
        f"Search box present: {search_box}\n"
        f"Visible buttons: {buttons}\n"
        f"Visible inputs: " + "; ".join(inputs)
    )


def normalize_locator(target: str) -> str:
    if not target:
        return ""

    target = target.strip()

   # tag:has-text("Text") or tag:has-text('Text') -> tag[text="Text"]
    m = re.match(r'^([a-zA-Z0-9]+):has-text\(["\'](.+)["\']\)$', target)
    if m:
        tag, text = m.groups()
        return f'{tag}[text="{text}"]'

    # keep already good attribute selectors unchanged
    if re.match(r'^[a-zA-Z0-9]+\[.+\]$', target):
        return target

    # button.Search -> button[class="Search"]
    m = re.match(r'^([a-zA-Z0-9]+)\.([a-zA-Z0-9_-]+)$', target)
    if m:
        tag, cls = m.groups()
        return f'{tag}[class="{cls}"]'

    # #search-btn -> *[id="search-btn"]
    m = re.match(r'^#([a-zA-Z0-9_-]+)$', target)
    if m:
        return f'*[id="{m.group(1)}"]'

    return target


def format_output(action):
    a_type = action.get("type", "").strip()
    target = normalize_locator(action.get("target", ""))
    value = str(action.get("value", "")).strip()

    if a_type == "TYPE":
        return f"TYPE {target} {value}".strip()
    elif a_type == "CLICK":
        return f"CLICK {target}".strip()
    elif a_type == "SELECT":
        return f"SELECT {target} {value}".strip()
    else:
        return json.dumps(action, ensure_ascii=False)


def convert_raw_to_sft(
    input_file="data/raw/raw_dataset_success_only.jsonl",
    output_file="data/processed/sft_data.jsonl"
):
    input_path = Path(input_file)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    print("Converting raw dataset to SFT format...")

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)

            sft_row = {
                "instruction": row.get("instruction", "").strip(),
                "input": format_input(row.get("observation_before", {})).strip(),
                "output": format_output(row.get("action", {})).strip()
            }

            fout.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Converted rows: {total}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    convert_raw_to_sft()