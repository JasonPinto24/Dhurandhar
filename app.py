def freshness_score(timestamp, emergency=False):
    """
    Returns freshness score using exponential decay.
    Handles:
    - ISO timestamps
    - ISO timestamps with timezone
    - UNIX timestamps
    - Missing / invalid timestamps
    """

    if not timestamp:
        return 1.0

    try:
        # Case 1: UNIX timestamp (int or float)
        if isinstance(timestamp, (int, float)):
            doc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Case 2: ISO string
        elif isinstance(timestamp, str):
            doc_time = datetime.fromisoformat(timestamp)

            # Force timezone-aware
            if doc_time.tzinfo is None:
                doc_time = doc_time.replace(tzinfo=timezone.utc)

        else:
            return 1.0

    except Exception:
        return 1.0

    now = datetime.now(timezone.utc)
    hours_old = (now - doc_time).total_seconds() / 3600

    lambda_val = 0.08 if emergency else 0.02
    return math.exp(-lambda_val * hours_old)
