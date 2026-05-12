from webarena_rl.env import WebArenaShoppingEnv


def test_bad_click_meta_filters_skip_links() -> None:
    env = WebArenaShoppingEnv("http://x", "http://y", max_steps=5, seed=1)
    assert env._bad_click_meta({"class_name": "skip-link", "text": "Skip to content", "href": "#main"}) is True
    assert env._bad_click_meta({"class_name": "", "text": "Product details", "href": "/p"}) is False
