import json
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

def collect_one_episode():
    episode = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        # Step 0: open page
        page.goto("http://localhost:7780")
        page.wait_for_timeout(3000)

        observation_before = {
            "url": page.url,
            "title": page.title(),
            "visible_buttons": [b.strip() for b in page.locator("button").all_text_contents() if b.strip()],
            "search_box_present": page.locator('input[name="q"]').count() > 0
        }

        action = {
            "type": "TYPE_AND_CLICK_SEARCH",
            "target": 'input[name="q"] + Search button',
            "value": "phone"
        }

        # Execute action
        page.fill('input[name="q"]', "phone")
        page.get_by_role("button", name="Search").click()
        page.wait_for_timeout(5000)

        observation_after = {
            "url": page.url,
            "title": page.title(),
            "visible_buttons": [b.strip() for b in page.locator("button").all_text_contents() if b.strip()]
        }

        step_record = {
            "task_id": "shopping_search",
            "episode_id": "episode_001",
            "step_id": 1,
            "instruction": "Search for a phone product",
            "observation_before": observation_before,
            "action": action,
            "observation_after": observation_after,
            "reward": 1 if "search" in page.url.lower() or "catalogsearch" in page.url.lower() else 0,
            "done": True,
            "timestamp": datetime.now().isoformat()
        }

        episode.append(step_record)

        browser.close()

    output_path = Path("data/raw/episode_001.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2)

    print(f"Saved episode to: {output_path}")

if __name__ == "__main__":
    collect_one_episode()