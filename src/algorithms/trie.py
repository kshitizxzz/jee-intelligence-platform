class TrieNode:
    __slots__ = ("children", "payloads")
    def __init__(self):
        self.children = {}
        self.payloads = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, text):
        node = self.root
        for ch in text.lower():
            node = node.children.setdefault(ch, TrieNode())
            node.payloads.append(text)

    def build(self, texts):
        for t in set(texts):
            self.insert(t)
        return self

    def autocomplete(self, prefix, limit=10):
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]
        seen = []
        for p in node.payloads:
            if p not in seen:
                seen.append(p)
            if len(seen) >= limit:
                break
        return seen
