"""Client for scripts/webshop_server.py (WebShop text env over HTTP).  Sessions index WebShop goals; split convention used here:
test = sessions 0-499, dev = 500-699, train = 700-2199."""
import os, requests
SERVER = os.environ.get("WEBSHOP_SERVER", "http://acn216:38300")
SPLITS = {"test": range(0, 500), "dev": range(500, 700), "train": range(700, 2200)}


class WebShopEnv:
    def __init__(self, worker="w0", server=SERVER): self.w = worker; self.s = server

    def reset(self, session):
        r = requests.post(f"{self.s}/reset", json={"worker": self.w, "session": int(session)}, timeout=300).json()
        self.actions = r["actions"]; return r["obs"], r["instruction"]

    def step(self, action):
        r = requests.post(f"{self.s}/step", json={"worker": self.w, "action": action}, timeout=300).json()
        self.actions = r["actions"]; return r["obs"], r["reward"], r["done"]

    def valid_actions(self):
        """click[...] for every clickable; search is free-form (signalled by has_search_bar)."""
        return [f"click[{c}]" for c in self.actions["clickables"]], bool(self.actions["has_search_bar"])

    def goal(self, session): return requests.get(f"{self.s}/goal", params={"session": int(session)}, timeout=300).json()

    def oracle(self, session, budget=30):
        return requests.post(f"{self.s}/oracle", json={"worker": self.w, "session": int(session), "budget": budget}, timeout=600).json()
