from datetime import datetime
from pathlib import Path
import sys
import json

from playwright.sync_api import sync_playwright

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.logger import save_episode
from src.env.browser_env import extract_observation
from src.agent.action_schema import execute_action
from src.agent.baseline_agent import choose_action


search_tasks = [
    "Search for a bag",
    "Search for a shirt",
    "Search for shoes",
    "Search for a watch",
    "Search for headphones",
    "Search for a backpack",
    "Search for a bottle",
    "Search for a keyboard",
    "Search for a mouse",
    "Search for a phone",
    "Search for a laptop",
    "Find a bag",
    "Find a shirt",
    "Find shoes",
    "Find a watch",
    "Find headphones",
    "Find a backpack",
    "Find a bottle",
    "Find a keyboard",
    "Find a mouse",
    "Look for a bag",
    "Look for a shirt",
    "Look for shoes",
    "Look for a watch",
    "Look for headphones",
    "Look for a backpack",
    "Look for a bottle",
    "Look for a keyboard",
    "Look for a mouse",
]

add_to_cart_tasks = [
    "Search for a bag and add it to cart",
    "Search for a shirt and add it to cart",
    "Search for shoes and add it to cart",
    "Search for a watch and add it to cart",
    "Search for headphones and add it to cart",
    "Search for a backpack and add it to cart",
    "Search for a bottle and add it to cart",
    "Search for a keyboard and add it to cart",
    "Search for a mouse and add it to cart",
    "Search for a phone and add it to cart",
    "Find a bag and add it to the cart",
    "Find a shirt and add it to the cart",
    "Find shoes and add them to the cart",
    "Find a watch and add it to the cart",
    "Find headphones and add them to the cart",
    "Find a backpack and add it to the cart",
    "Find a bottle and add it to the cart",
    "Find a keyboard and add it to the cart",
    "Find a mouse and add it to the cart",
    "Find a laptop and add it to the cart",
    "Look for a bag and add it to cart",
    "Look for a shirt and add it to cart",
    "Look for shoes and add them to cart",
    "Look for a watch and add it to cart",
    "Look for headphones and add them to cart",
    "Look for a backpack and add it to cart",
    "Look for a bottle and add it to cart",
    "Look for a keyboard and add it to cart",
    "Look for a mouse and add it to cart",
]

navigation_tasks = [
    "Go to the Sign In page",
    "Open the Cart page",
    "Navigate to the Home page",
]

all_tasks = search_tasks + add_to_cart_tasks + navigation_tasks
RAW_PATH = "data/raw/raw_dataset_success_only.jsonl"


def get_task_family(instruction):
    if instruction in search_tasks:
        return "search"
    elif instruction in add_to_cart_tasks:
        return "add_to_cart"
    elif instruction in navigation_tasks:
        return "navigation"
    else:
        return "unknown"


def compute_reward(page, instruction, task_family):
    url_lower = page.url.lower()
    instruction_lower = instruction.lower()

    if task_family == "search":
        return 1 if "catalogsearch" in url_lower else 0

    elif task_family == "navigation":
        if "sign in" in instruction_lower:
            return 1 if ("login" in url_lower or "account" in url_lower) else 0
        elif "cart" in instruction_lower:
            return 1 if "checkout/cart" in url_lower else 0
        elif "home" in instruction_lower:
            return 1 if url_lower.rstrip("/") == "http://localhost:7770" else 0
        return 0

    elif task_family == "add_to_cart":
        page_text = page.locator("body").inner_text().lower()
        return 1 if (
            "shopping cart" in page_text
            or "was added to your shopping cart" in page_text
            or "checkout/cart" in url_lower
        ) else 0

    return 0


def count_existing_successes(path):
    file_path = Path(path)
    if not file_path.exists():
        return 0, 1

    successful_episodes = set()
    max_episode_num = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            episode_id = row.get("episode_id", "")
            if row.get("done") is True and row.get("reward") == 1:
                successful_episodes.add(episode_id)

            if episode_id.startswith("episode_"):
                try:
                    num = int(episode_id.split("_")[1])
                    max_episode_num = max(max_episode_num, num)
                except ValueError:
                    pass

    return len(successful_episodes), max_episode_num + 1


def run_single_episode(page, episode_number, max_steps=6):
    episode = []
    episode_id = f"episode_{episode_number:03d}"

    instruction = all_tasks[(episode_number - 1) % len(all_tasks)]
    task_family = get_task_family(instruction)

    print("Running task:", instruction)

    page.goto("http://localhost:7770")
    page.wait_for_timeout(2000)

    success = False

    for step_id in range(1, max_steps + 1):
        observation_before = extract_observation(page)
        action = choose_action(observation_before, instruction)

        execute_action(page, action)
        page.wait_for_timeout(2000)

        observation_after = extract_observation(page)
        reward = compute_reward(page, instruction, task_family)
        done = action["type"] == "STOP" or reward == 1

        step_record = {
            "task_id": task_family,
            "episode_id": episode_id,
            "step_id": step_id,
            "instruction": instruction,
            "observation_before": observation_before,
            "action": action,
            "observation_after": observation_after,
            "reward": reward,
            "done": done,
            "timestamp": datetime.now().isoformat()
        }

        episode.append(step_record)

        if reward == 1:
            success = True

        if done:
            break

    if success and episode and episode[-1]["reward"] == 1:
        save_episode(episode, RAW_PATH)
        print(f"Saved successful episode: {episode_id}")
        return True
    else:
        print(f"Discarded unsuccessful episode: {episode_id}")
        return False


def collect_until_success_target(target_successes=500):
    existing_successes, next_episode_number = count_existing_successes(RAW_PATH)
    success_count = existing_successes
    episode_number = next_episode_number

    print(f"Existing successful trajectories: {existing_successes}")
    print(f"Starting from episode number: {episode_number}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        while success_count < target_successes:
            print(f"Running episode {episode_number}...")
            page = browser.new_page()

            try:
                was_successful = run_single_episode(page, episode_number)
                if was_successful:
                    success_count += 1
                    print(f"Successful trajectories collected: {success_count}/{target_successes}")
            except Exception as e:
                print(f"Error in episode {episode_number}: {e}")
            finally:
                page.close()

            episode_number += 1

        browser.close()


if __name__ == "__main__":
    collect_until_success_target(target_successes=500)