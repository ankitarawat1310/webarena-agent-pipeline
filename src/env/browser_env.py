def extract_observation(page):
    buttons = [b.strip() for b in page.locator("button").all_text_contents() if b.strip()]

    inputs = page.locator("input").evaluate_all(
        """elements => elements.map(el => ({
            type: el.type || "",
            name: el.name || "",
            placeholder: el.placeholder || "",
            value: el.value || ""
        }))"""
    )

    return {
        "url": page.url,
        "title": page.title(),
        "visible_buttons": buttons,
        "visible_inputs": inputs,
        "search_box_present": page.locator('input[name="q"]').count() > 0
    }