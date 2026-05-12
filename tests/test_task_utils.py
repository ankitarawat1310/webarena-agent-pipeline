from webarena_rl.task_utils import extract_search_query, parse_task_spec


def test_task_query_and_price_extraction() -> None:
    spec = parse_task_spec("Find a wireless mouse under $50 and add it to cart.")
    assert spec.product_query == "wireless mouse"
    assert spec.max_price == 50.0
    assert spec.require_cart_or_checkout is True
    assert "wireless mouse" in spec.product_keywords
    assert "wireless" in spec.product_keywords


def test_task_parser_variants() -> None:
    spec = parse_task_spec("Search for gaming keyboard under $120 and open product details.")
    assert spec.product_query == "gaming keyboard"
    assert spec.max_price == 120.0
    assert spec.require_cart_or_checkout is False
    assert extract_search_query(spec.raw_instruction) == "gaming keyboard"
