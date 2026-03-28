from playwright.sync_api import sync_playwright

def test_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("http://localhost:7780")
        page.wait_for_timeout(3000)

        print("Before action URL:", page.url)

        # Type in the search box
        page.fill('input[name="q"]', "phone")

        # Click the Search button
        page.get_by_role("button", name="Search").click()

        page.wait_for_timeout(5000)

        print("After action URL:", page.url)
        print("Page title:", page.title())

        input("\nPress ENTER to close browser...")
        browser.close()

if __name__ == "__main__":
    test_browser()