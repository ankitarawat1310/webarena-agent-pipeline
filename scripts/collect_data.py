from datetime import datetime
from pathlib import Path
import sys
import random

from playwright.sync_api import sync_playwright

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.logger import save_episode
from src.env.browser_env import extract_observation
from src.agent.action_schema import execute_action
from src.agent.baseline_agent import choose_action


SEARCH_TASKS = [
    "Search for a phone",
    "Search for a laptop",
    "Search for a shirt",
    "Search for a watch",
    "Search for a bag"
]


def run_single_episode(page, episode_number):
    episode = []
    instruction = random.choice(SEARCH_TASKS)
    episode_id = f"episode_{episode_number:03d}"

    page.goto("http://localhost:7780")
    page.wait_for_timeout(3000)

    observation_before = extract_observation(page)
    action = choose_action(observation_before, instruction)

    execute_action(page, action)
    page.wait_for_timeout(5000)

    observation_after = extract_observation(page)

    reward = 1 if "search" in page.url.lower() or "catalogsearch" in page.url.lower() else 0
    done = True if action["type"] != "STOP" else False

    step_record = {
        "task_id": "shopping_search",
        "episode_id": episode_id,
        "step_id": 1,
        "instruction": instruction,
        "observation_before": observation_before,
        "action": action,
        "observation_after": observation_after,
        "reward": reward,
        "done": done,
        "timestamp": datetime.now().isoformat()
    }

    episode.append(step_record)
    save_episode(episode, f"data/raw/{episode_id}.json")


def collect_episode_range(start_episode=1, end_episode=10):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        for episode_number in range(start_episode, end_episode + 1):
            print(f"Running episode {episode_number}...")
            try:
                run_single_episode(page, episode_number)
            except Exception as e:
                print(f"Error in episode {episode_number}: {e}")

        browser.close()


if __name__ == "__main__":
    collect_episode_range(start_episode=11, end_episode=49)