from collections import defaultdict


class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mp = defaultdict()

        if len(s) != len(t):
            return False


        for i in range(len(s)):
            if s[i] not in mp and t[i] not in mp:
                mp[s[i]] = t[i]
                mp[t[i]] = s[i]
            else:
                if mp[s[i]] != t[i]:
                    return False

        return True