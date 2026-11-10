from research_agent.net.ratelimit import DEFAULT_INTERVALS, HostLimits, RateLimiter

API_URL = "https://export.arxiv.org/api/query?search_query=all:speculative"
HTML_URL = "https://arxiv.org/html/2609.01234"


class FakeClock:
    """A clock that moves only when something sleeps on it or the test advances it; sleeps are recorded."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now, self.sleeps = start, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limiter(interval: float) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(interval, clock=clock, sleep=clock.sleep), clock


def host_limits(**kwargs) -> tuple[HostLimits, FakeClock]:
    clock = FakeClock()
    return HostLimits(clock=clock, sleep=clock.sleep, **kwargs), clock


def test_first_call_never_waits():
    lim, clock = limiter(15.0)
    assert lim.wait() == 0.0 and clock.sleeps == []


def test_back_to_back_calls_are_spaced_by_the_interval():
    lim, clock = limiter(3.0)
    returned = []
    for _ in range(3):
        lim.wait()
        returned.append(clock.now)
    assert clock.sleeps == [3.0, 3.0] and returned == [1000.0, 1003.0, 1006.0]


def test_time_spent_since_the_previous_call_counts():
    lim, clock = limiter(3.0)
    lim.wait()
    clock.advance(2.0)  # the request itself took 2 s
    assert lim.wait() == 1.0
    clock.advance(10.0)
    assert lim.wait() == 0.0
    assert clock.sleeps == [1.0]  # no zero-length sleeps


def test_interval_runs_from_when_the_previous_wait_returned():
    lim, clock = limiter(15.0)
    lim.wait()  # returns at 1000
    clock.advance(1.0)
    assert lim.wait() == 14.0  # returns at 1015
    clock.advance(0.5)
    assert lim.wait() == 14.5
    assert clock.sleeps == [14.0, 14.5]


def test_default_intervals_are_the_arxiv_limits_per_host():
    assert DEFAULT_INTERVALS == {"export.arxiv.org": 3.0, "arxiv.org": 15.0}
    lim, clock = host_limits()
    for url in (API_URL, HTML_URL, API_URL, HTML_URL):
        lim.wait(url)
    # the second API call waits 3 s; by then 3 of the 15 s between the HTML calls have passed
    assert clock.sleeps == [3.0, 12.0]


def test_other_hosts_get_the_default_interval_each_on_its_own():
    lim, clock = host_limits()
    lim.wait("https://example.org/a")
    lim.wait("https://example.org/b")
    assert clock.sleeps == [1.0]
    lim, clock = host_limits(default=2.0)
    lim.wait("https://a.example.org/x")
    lim.wait("https://b.example.org/y")  # another host: its first call
    lim.wait("https://a.example.org/z")
    assert clock.sleeps == [2.0]


def test_hosts_are_matched_case_insensitively_and_without_the_port():
    lim, clock = host_limits(intervals={"Example.ORG": 5.0})
    lim.wait("https://EXAMPLE.org/a")
    lim.wait("https://example.org:443/b")
    assert clock.sleeps == [5.0]


def test_a_host_written_with_its_root_dot_is_the_same_host():
    lim, clock = host_limits()
    lim.wait(HTML_URL)
    lim.wait("https://arxiv.org./html/2609.01234v2")  # e.g. a redirect hop; httpx keeps the trailing dot
    assert clock.sleeps == [15.0]
    lim, clock = host_limits(intervals={"example.org.": 5.0})
    lim.wait("https://example.org/a")
    lim.wait("https://example.org/b")
    assert clock.sleeps == [5.0]


def test_explicit_intervals_replace_the_defaults():
    lim, clock = host_limits(intervals={}, default=0.5)
    lim.wait(HTML_URL)
    lim.wait(HTML_URL)
    assert clock.sleeps == [0.5]
