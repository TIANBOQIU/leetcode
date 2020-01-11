## LeetCode is Simple

### Part 1

#### [42] Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/

```java
class Solution {
    public int trap(int[] height) {
        if (height.length < 3) return 0;
        int l = 0, r = height.length - 1, water = 0;
        while (l < r) {
            int left = height[l], right = height[r]; // we need to count from the shorter side
            if (left <= right) {
                while (l < r && height[++l] <= left) {
                    water += left - height[l];
                }
            } else {
                while (l < r && height[--r] <= right) {
                    water += right - height[r];
                }
            }
        }
        return water;
    }
}
```

#### Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);
        int l = 0, r = nums1.length;
        while (l <= r) {
            int mid = (l + r) / 2;
            int len1 = mid, len2 = (nums1.length + nums2.length + 1) / 2 - len1; // make mid on the left part
            int left1 = (len1 == 0 ? Integer.MIN_VALUE : nums1[len1 - 1]);
            int right1 = (len1 == nums1.length ? Integer.MAX_VALUE : nums1[len1]);
            int left2 = (len2 == 0 ? Integer.MIN_VALUE : nums2[len2 - 1]);
            int right2 = (len2 == nums2.length ? Integer.MAX_VALUE : nums2[len2]);
            if (left1 <= right2 && left2 <= right1) {
                if ((nums1.length + nums2.length) % 2 == 1) {
                    return Math.max(left1, left2);
                } else {
                    return ((double) (Math.max(left1, left2)) + (double) (Math.min(right1, right2))) / 2;
                }
            } else if (left1 > right2) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return -1;
    }
    // search for the length of left part of the shorter array, the left part of nums1 and the left part of nums 2 sum to the combined left part
}
```

#### [1192] Critical Connections in a Network

https://leetcode.com/problems/critical-connections-in-a-network/

> find bridges, Tarjan Algorithm

```java
class Solution {
    int cnt = 0;
    int[] ids;
    int[] low;
    List<List<Integer>> bridges = new ArrayList<>();
    public List<List<Integer>> criticalConnections(int n, List<List<Integer>> connections) {
        ids = new int[n]; Arrays.fill(ids, -1); // init val matters
        low = new int[n]; Arrays.fill(low, -1);
        Set<Integer>[] G = new Set[n]; // graph
        for (int i = 0; i < n; i++) G[i] = new HashSet<>(); // populate nodes
        for (List<Integer> e : connections) { // populate edges
            G[e.get(0)].add(e.get(1));
            G[e.get(1)].add(e.get(0));
        }
        for (int i = 0; i < n; i++) {
            if (ids[i] == -1) dfs(G, i, i);
        }
        return bridges;
    }

    private void dfs(Set<Integer>[] G, int i, int parent) {
        ids[i] = low[i] = ++cnt;
        for (int to : G[i]) {
            if (to == parent) continue;
            if (ids[to] != -1) low[i] = Math.min(low[i], low[to]); // visited before
            else {
                dfs(G, to, i);
                if (low[to] > ids[i]) bridges.add(Arrays.asList(i, to));
                low[i] = Math.min(low[i], low[to]);
            }
        }
    }

}
```

```
nodes in cycle will have the same low val
> System.out.println(Arrays.toString(ids));
> System.out.println(Arrays.toString(low));
for the graph in the description
std output:
[1, 2, 3, 4]
[1, 1, 1, 4]
```

#### Merge k Sorted Lists

https://leetcode.com/problems/merge-k-sorted-lists/

```java
// priority queue solution, O(k) in space, O(Nlogk) in time

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((l1, l2) -> (l1.val - l2.val));
        for (ListNode l : lists)
            if (l != null) pq.offer(l); // check null
        ListNode head = new ListNode(-1), p = head;
        while (!pq.isEmpty()) {
            ListNode cur = pq.poll();
            p.next = cur; p = p.next;
            if (cur.next != null) pq.offer(cur.next);
        }
        return head.next;
    }
}
```

```java
// merge sort, O(1) in space, O(Nlogk) in time
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        return helper(lists, 0, lists.length - 1);
    }

    private ListNode helper(ListNode[] lists, int start, int end) {
        if (start > end) return null;
        if (start == end) return lists[start];
        int mid = (start + end) / 2;
        return merge(helper(lists, start, mid), helper(lists, mid + 1, end));
    }

    private ListNode merge(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) return l1 != null ? l1 : l2;
        if (l1.val < l2.val) {
            l1.next = merge(l1.next, l2);
            return l1;
        } else {
            l2.next = merge(l1, l2.next);
            return l2;
        }
    }
}
```

```cpp
// ref from grandyang
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        return helper(lists, 0, (int)lists.size() - 1);
    }
    ListNode* helper(vector<ListNode*>& lists, int start, int end) {
        if (start > end) return NULL;
        if (start == end) return lists[start];
        int mid = start + (end - start) / 2;
        ListNode *left = helper(lists, start, mid);
        ListNode *right = helper(lists, mid + 1, end);
        return mergeTwoLists(left, right);
    }
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (!l1) return l2;
        if (!l2) return l1;
        if (l1->val < l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
```

#### [76] Minimum Window Substring

https://leetcode.com/problems/minimum-window-substring/

> move j until find all, then shrink i and update min until the condition is not met

```java
class Solution {
    public String minWindow(String s, String t) {
        int[] count = new int[128]; // if ASCII
        for (char ch : t.toCharArray()) count[ch]++;
        int i = 0, cnt = 0, min = Integer.MAX_VALUE, minIndex = -1;
        for (int j = 0; j < s.length(); j++) {
            char ch = s.charAt(j);
            if (--count[ch] >= 0) cnt++;
            while (cnt == t.length()) {
                if (j - i + 1 < min) {
                    min = j - i + 1;
                    minIndex = i;
                }
                if (++count[s.charAt(i++)] > 0) --cnt;
                //i++;
            }
        }
        return minIndex == -1 ? "" : s.substring(minIndex, minIndex + min);
    }
}
```

#### [273] Integer to English Words

https://leetcode.com/problems/integer-to-english-words/

```java
class Solution {
    String[] less_than_20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    String[] tens = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    String[] thousands = {"", "Thousand", "Million", "Billion"};
    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        String res = "";
        int i = 0;
        while (num > 0) {
            if (num % 1000 != 0) {
                res = helper(num % 1000) + thousands[i] + " " +res;
            }
            i++;
            num /= 1000;
        }
        return res.trim();
    }

    private String helper(int n) {
        if (n == 0) return "";
        if (n < 20) return less_than_20[n] + " ";
        if (n < 100) return tens[n / 10] + " " + helper(n % 10);
        return less_than_20[n / 100] + " Hundred " + helper(n % 100);
    }
}
```

#### [301] Remove Invalid Parentheses

https://leetcode.com/problems/remove-invalid-parentheses/

> first count how many open and close parentheses we need to remove, actually its already the minimum number we need to remove to make the string valid

> then dfs

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        int open = 0, close = 0;
        for (char ch : s.toCharArray()) {
            if (ch == '(') open++;
            if (ch == ')') {
                if (open > 0) open--;
                else close++;
            }
        }
        List<String> res = new ArrayList<>();
        dfs(s, 0, open, close, res);
        return res;
    }

    private void dfs(String s, int start, int open, int close, List<String> res) {
        if (open < 0 || close < 0) return;
        if (open == 0 && close == 0 && check(s)) {
            res.add(s); return;
        }
        for (int i = start; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (i > start && s.charAt(i) == s.charAt(i - 1)) continue; // remove dups
            if (ch == '(' || ch == ')') {
                String next = s.substring(0, i) + s.substring(i + 1);
                if (ch == '(' && open > 0)
                    dfs(next, i, open - 1, close, res); // start from i
                if (ch == ')' && close > 0)
                    dfs(next, i, open, close - 1, res);
            }
        }
    }

    private boolean check(String s) {
        int cnt = 0;
        for (char ch : s.toCharArray()) {
            if (ch == '(') cnt++;
            if (ch == ')') cnt--;
            if (cnt < 0) return false;
        }
        return cnt == 0;
    }
}
```

#### [297] Serialize and Deserialize Binary Tree

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb);
        return sb.toString();
    }

    private void serialize(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append("X").append(",");
            return;
        }
        sb.append(root.val).append(",");
        serialize(root.left, sb);
        serialize(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>();
        queue.addAll(Arrays.asList(data.split(",")));
        return deserialize(queue);
    }

    private TreeNode deserialize(Queue<String> queue) {
        String cur = queue.poll();
        if (cur.equals("X")) return null;
        TreeNode root = new TreeNode(Integer.valueOf(cur));
        root.left = deserialize(queue);
        root.right = deserialize(queue);
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```

#### [269] Alien Dictionary

https://leetcode.com/problems/alien-dictionary

**topological order**

> DFS with return value (contains cycle or not) and postorder traversal to get the topological order

> or BFS, use Kahn' s algorithm

```java
class Solution {
    public String alienOrder(String[] words) {
        boolean[][] adj = new boolean[26][26];
        int[] visited = new int[26];
        Arrays.fill(visited, -1); // -1 unvisited, 0 not visited, 1 visiting, 2 visited
        for (int i = 0; i < words.length; i++) {
            for (char ch : words[i].toCharArray()) visited[ch - 'a'] = 0;
            if (i > 0) {
                String w1 = words[i - 1], w2 = words[i];
                for (int j = 0; j < Math.min(w1.length(), w2.length()); j++) {
                    char ch1 = w1.charAt(j), ch2 = w2.charAt(j);
                    if (ch1 != ch2) {
                        adj[ch1 - 'a'][ch2 - 'a'] = true;
                        break;
                    }
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 26; i++) {
            if (visited[i] == 0) {
                if (dfs(adj, visited, i, sb)) return "";
            }
        }
        return sb.reverse().toString();

    }

    private boolean dfs(boolean[][] adj, int[] visited, int i, StringBuilder sb) {
        visited[i] = 1;
        for (int j = 0; j < 26; j++) { // edge ['a' + i] -> ['a' + j]
            if (!adj[i][j]) continue;
            if (visited[j] == 1) return true; // contains cycle
            if (visited[j] == 0 && dfs(adj, visited, j, sb)) return true;
        }
        visited[i] = 2;
        sb.append((char)('a' + i));
        return false; // no cycle
    }
}
```

```java
// BFS, kahn
class Solution {
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> G = new HashMap<>();
        Map<Character, Integer> degree = new HashMap<>();
        for (String w : words) {
            for (char ch : w.toCharArray()) { // populate node
                if (!G.containsKey(ch)) G.put(ch, new HashSet<>());
                if (!degree.containsKey(ch)) degree.put(ch, 0);
            }
        }
        for (int i = 1; i < words.length; i++) { // populate edges
            String w1 = words[i - 1], w2 = words[i];
            for (int j = 0; j < Math.min(w1.length(), w2.length()); j++) {
                char ch1 = w1.charAt(j), ch2 = w2.charAt(j);
                if (ch1 != ch2) {
                    if (!G.get(ch1).contains(ch2)) { // dup edges
                        G.get(ch1).add(ch2);
                        degree.put(ch2, degree.getOrDefault(ch2, 0) + 1);
                    }
                    break;
                }
            }
        }
        StringBuilder res = new StringBuilder(); // topo, kahn
        Queue<Character> queue = new LinkedList<>();
        for (char ch : degree.keySet())
            if (degree.get(ch) == 0) queue.offer(ch);
        while (!queue.isEmpty()) {
            char ch = queue.poll();
            res.append(ch);
            for (char c : G.get(ch)) {
                degree.put(c, degree.get(c) - 1);
                if (degree.get(c) == 0) queue.offer(c);
            }
        }
        System.out.println(res.toString());
        return res.length() == degree.size() ? res.toString() : "";
    }
}
```

> another implementation of bfs using adj, there is an edge case

```
[za, zb, ca, cb]

edge a->b appears twice, the degree of b becomes 2
```

```java
class Solution {
    public String alienOrder(String[] words) {
        boolean[][] adj = new boolean[26][26];
        int[] degree = new int[26];
        Arrays.fill(degree, -1);
        for (int i = 0; i < words.length; i++) {
            for (char ch : words[i].toCharArray()) {
                if (degree[ch - 'a'] == -1) degree[ch - 'a'] = 0;
            }
            if (i > 0) {
                String w1 = words[i - 1], w2 = words[i];
                for (int j = 0; j < Math.min(w1.length(), w2.length()); j++) {
                    char ch1 = w1.charAt(j), ch2 = w2.charAt(j);
                    if (ch1 != ch2) {
                        if (!adj[ch1 - 'a'][ch2 - 'a']) degree[ch2 - 'a']++; // avoid duplicate edges
                        adj[ch1 - 'a'][ch2 - 'a'] = true;
                        //degree[ch2 - 'a']++;
                        break;
                    }
                }
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        int cnt = 0;
        for (int i = 0; i < 26; i++) {
            if (degree[i] != -1) cnt++;
            if (degree[i] == 0) queue.offer(i);
        }
        while (!queue.isEmpty()) {
            int i = queue.poll();
            sb.append((char)('a' + i));
            for (int j = 0; j < 26; j++) {
                if (adj[i][j] && --degree[j] == 0) queue.offer(j);
            }
        }
        return sb.length() == cnt ? sb.toString() : "";
    }
}
```

#### [68] Text Justification

https://leetcode.com/problems/text-justification/

```java
class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        int i = 0, len = 0, n = words.length;
        for (int j = 0; j <= n; j++) {
            if (j == n || len + words[j].length() + j - i > maxWidth) { // format [i, j)
                int slots = j - 1 - i, space = maxWidth - len;
                StringBuilder sb = new StringBuilder();
                if (j == n || slots == 0) { // left align
                    for (int k = i; k < j; k++) {
                        sb.append(words[k]);
                        if (k != j - 1) sb.append(" ");
                    }
                    int last = maxWidth - sb.length();
                    while (last -- > 0) sb.append(" ");
                } else {
                    int extra = space % slots;
                    for (int k = i; k < j; k++) {
                        sb.append(words[k]);
                        if (k != j - 1) {
                            int between = space / slots;
                            while (between-- > 0) sb.append(" ");
                            if (extra-- > 0) sb.append(" ");
                        }
                    }
                }
                res.add(sb.toString());
                i = j; len = 0; // reset
            }
            if (j < n) len += words[j].length();
        }
        return res;
    }
}
```

#### [336] Palindrome Pairs

https://leetcode.com/problems/palindrome-pairs/

```java
class Solution {
    public List<List<Integer>> palindromePairs(String[] words) {
        Set<List<Integer>> res = new HashSet<>();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < words.length; i++)
            map.put(words[i], i);
        for (int i = 0; i < words.length; i++) {
            for (int j = 0; j <= words[i].length(); j++) {
                String s1 = words[i].substring(0, j), s2 = words[i].substring(j);
                if (check(s1)) {
                    String rev = new StringBuilder(s2).reverse().toString();
                    if (map.containsKey(rev) && map.get(rev) != i)
                        res.add(Arrays.asList(map.get(rev), i));
                }
                if (check(s2)) {
                    String rev = new StringBuilder(s1).reverse().toString();
                    if (map.containsKey(rev) && map.get(rev) != i)
                        res.add(Arrays.asList(i, map.get(rev)));
                }
            }
        }
        return new ArrayList<>(res);
    }

    private boolean check(String s) { // is palindormic
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) return false;
        }
        return true;
    }
}
```

### [829] Consecutive Numbers Sum

```
we can prove that 1 <= k < sqrt(2N) + 1

suppose the seq starts at x and has length k, then the sum is (2x + (k - 1)k) / 2 = N, so that (k - 1)^2 < (k - 1)k < 2N, thus k < sqrt(2N) + 1

and also x := N / k - (k - 1) / 2 = (2N - k(k-1)) / 2k
```

```java
class Solution {
    public int consecutiveNumbersSum(int N) {
        int cnt = 0;
        for (int k = 1; k < Math.sqrt(2 * N) + 1; k++) {
            if ((2 * N - k * (k - 1)) % (2 * k) == 0 && (2 * N - k * (k - 1)) / (2 * k) >= 1)
                cnt++;
        }
        return cnt;
    }
}
```

### [547] Friend Circles

> use dfs to mark all connected componet one by one, note that M[i][i] is set to 1 in the problem

```java
class Solution {
    public int findCircleNum(int[][] M) {
        int n = M.length, cnt = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs(i, M, visited);
                cnt++;
            }

        }
        return cnt;

    }

    private void dfs(int i, int[][] adj, boolean[] visited) {
        if (visited[i]) return;
        visited[i] = true;
        for (int j = 0; j < adj[i].length; j++)
            if (adj[i][j] == 1 && !visited[j])
                dfs(j, adj, visited);
    }
}
```

### [1048] Longest String Chain

```java
class Solution {
    public int longestStrChain(String[] words) {
        Map<String, Integer> dp = new HashMap<>();
        Arrays.sort(words, (s1, s2) -> (s1.length() - s2.length()));
        int max = 0;
        for (String s : words) {
            int cur = 1;
            for (int i = 0; i < s.length(); i++) {
                String prev = s.substring(0, i) + s.substring(i + 1);
                cur = Math.max(cur, dp.getOrDefault(prev, 0) + 1);
            }
            dp.put(s, cur);
            max = Math.max(max, cur);
        }
        return max;
    }
}
```

### [780] Reaching Points

> go backward from the target

```java
class Solution {
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        while (tx > sx && ty > sy) { // we can still go backward from t
            if (tx > ty) tx %= ty;
            else ty %= tx;
        }
        return (sx == tx && sy <= ty && (ty - sy) % sx == 0) ||
            (sy == ty && sx <= tx && (tx - sx) % sy == 0);
    }
}
```

### [157] Read N Characters Given Read4

> read n characters into the buf array and return the actual readed length, the goal is to read into the destination buffer

```java
/**
 * The read4 API is defined in the parent class Reader4.
 *     int read4(char[] buf);
 */
public class Solution extends Reader4 {
    /**
     * @param buf Destination buffer
     * @param n   Number of characters to read
     * @return    The number of actual characters read
     */
    public int read(char[] buf, int n) {
        int cnt = 0;
        boolean eof = false;
        while (!eof && cnt < n) {
            char[] tmp = new char[4];
            int len = read4(tmp);
            eof = len < 4;
            for (int i = 0; i < len && cnt < n; i++)
                buf[cnt++] = tmp[i];
        }
        return cnt;
    }
}
```

### [158]

```java
/**
 * The read4 API is defined in the parent class Reader4.
 *     int read4(char[] buf);
 */
public class Solution extends Reader4 {
    /**
     * @param buf Destination buffer
     * @param n   Number of characters to read
     * @return    The number of actual characters read
     */
    char[] buf4 = new char[4];
    int i4 = 0, n4 = 0;
    public int read(char[] buf, int n) {
        int i = 0;
        while (i < n) {
            if (i4 >= n4) { // inner buffer is empty, read4 again
                i4 = 0;
                n4 = read4(buf4);
                if (n4 == 0) break;
            }
            buf[i++] = buf4[i4++];
        }
        return i;
    }
}
```

### [218] The Skyline Problem

> TreeMap to handle duplicate keys, use the value as count. TreeMap supports logn insertion and lookup

> sort and sweep

```java
class Solution {
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<int[]> lines = new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        for (int[] b : buildings) {
            lines.add(new int[]{b[0], -b[2]}); // entering event, negative
            lines.add(new int[]{b[1], b[2]}); // leaving
        }
        Collections.sort(lines, (l1, l2) -> (l1[0] == l2[0] ? l1[1] - l2[1] : l1[0] - l2[0])); // enter high to low, leave l to h
        TreeMap<Integer, Integer> map = new TreeMap<>(); // treemap with count
        map.put(0, 1);
        int prev = 0;
        for (int[] l : lines) {
            if (l[1] < 0) {
                map.put(-l[1], map.getOrDefault(-l[1], 0) + 1);
            } else {
                map.put(l[1], map.getOrDefault(l[1], 0) - 1);
                if (map.get(l[1]) <= 0) map.remove(l[1]);
            }
            int cur = map.lastKey();
            if (cur != prev) {
                res.add(new ArrayList<>(Arrays.asList(l[0], cur)));
                prev = cur;
            }
        }
        return res;
    }
}
```

### [680] Vald Palindrome II

```java
class Solution {
    public boolean validPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) return check(s, i + 1, j) || check(s, i, j - 1);
            else {
                i++; j--;
            }
        }
        return true;
    }

    private boolean check(String s, int i, int j) {
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) return false;
            i++; j--;
        }
        return true;
    }
}
```

### [443] String Compression

```
note that rev of 10 becomes 1 not 01 as expected
```

```java
class Solution {
    public int compress(char[] chars) {
        if (chars.length == 0) return 0;
        char cur = chars[0];
        int cnt = 1, pos = 0;
        for (int i = 1; i <= chars.length; i++) {
            if (i == chars.length || chars[i] != cur) {
                chars[pos++] = cur;
                if (cnt > 1) {
                    for (char ch : Integer.toString(cnt).toCharArray())
                        chars[pos++] = ch;
                }
                if (i < chars.length) {
                    cur = chars[i];
                    cnt = 1;
                }
            } else cnt++;
        }
        return pos;
    }
}
```

### [93] Restore IP Addresses

```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        restore(s, 4, new StringBuilder(), res);
        return res;
    }

    public void restore(String s, int k, StringBuilder sb, List<String> res) {
        if (s.length() > 12) return;
        if (k == 0 && s.length() == 0) res.add(sb.toString());
        else {
            for (int i = 1; i <= 3 && i <= s.length(); i++) {
                if (isValid(s.substring(0, i))) {
                    String ip = s.substring(0, i);
                    sb.append(ip); if (k > 1) sb.append('.');
                    restore(s.substring(i), k - 1, sb, res);
                    if (k > 1) sb.deleteCharAt(sb.length() - 1);
                    sb.delete(sb.length() - ip.length(), sb.length());
                }
            }
        }
    }

    public boolean isValid(String s) {
        int n = s.length();
        if (n <= 0 || n > 3 || (n > 1 && s.charAt(0) == '0')) return false;
        int ip = Integer.valueOf(s);
        return ip >= 0 && ip <= 255;
    }
}
```

> I use StringBuilder here, but I think I can just use string since the ip string is short

### [706] Design HashMap

1. Are the kes integers only?
2. Can we use chaining to solve collision
3. Do I need to consider the load factor / resizing?

> There are numbers of different solutions when I saw this problem. Note that the keys and vallues have been limited into [0, 1,000,000]

#### solution 1, use an array of size 1e6

> we directly put the value into the index which equals to the key

```java
class MyHashMap {
    int[] arr;
    /** Initialize your data structure here. */
    public MyHashMap() {
        arr = new int[1000000];
        Arrays.fill(arr, -1);
    }

    /** value will always be non-negative. */
    public void put(int key, int value) {
        arr[key] = value;
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        return arr[key];
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        arr[key] = -1;
    }
}
```

#### optimization 1: dynamic size

> we don't have to allocate an array of size 1e6 initially. We can allocate an array of 1000 and then allocate a subarray of size 1000 when needed. For example, key := 100002 will be mapped into arr[10002 % 2][10002 / 1000] which is something like arr[last 3 digit][first 3 digit]

```java
class MyHashMap {
    int[][] arr;
    /** Initialize your data structure here. */
    public MyHashMap() {
        arr = new int[1000][]; // arr[0] == null
    }

    /** value will always be non-negative. */
    public void put(int key, int value) {
        int hash = key % 1000;
        if (arr[hash] == null) {
            arr[hash] = new int[1000];
            Arrays.fill(arr[hash], -1);
        }
        arr[hash][key / 1000] = value;
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        int hash = key % 1000;
        if (arr[hash] == null) return -1;
        return arr[hash][key / 1000];
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        int hash = key % 1000;
        if (arr[hash] != null) {
            arr[hash][key / 1000] = -1;
        }
    }
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.put(key,value);
 * int param_2 = obj.get(key);
 * obj.remove(key);
 */
```

#### optimization 2: for the sub array, we can also use a linkedlist instead of an array of size 1000 to save some space, but we have to gp through the list for the (key, value) pair which takes more time than the second solution

> use linkedlist or ~~any dynamic sized array (like arraylist or vector in c++)~~(we need to chain the key value pairs) to further optimize the space, but will take more time for the get operations, a space/time tradeoff

> the best so far

```java
class MyHashMap {
    Node[] arr;
    /** Initialize your data structure here. */
    public MyHashMap() {
        arr = new Node[1000];
    }

    /** value will always be non-negative. */
    public void put(int key, int value) {
        int hash = key % 1000; // we can also use Integer.hashCode(key) % arr.length, etc
        if (arr[hash] == null) arr[hash] = new Node();
        Node node = arr[hash];
        while (node.key != key && node.next != null) node = node.next;
        if (node.key == key) node.val = value;
        else node.next = new Node(key, value);

    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        int hash = key % 1000;
        if (arr[hash] == null) return -1;
        Node node = arr[hash];
        while (node.key != key && node.next != null) node = node.next;
        return node.key == key ? node.val : -1;
    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void remove(int key) {
        int hash = key % 1000;
        if (arr[hash] == null) return;
        Node node = arr[hash], pre = node;
        while(node.key != key && node.next != null) {
            pre = node;
            node = node.next;
        }
        if (node.key == key) pre.next = node.next;
    }

    private class Node {
        int key = -1;
        int val = -1;
        Node next = null;
        public Node() {};
        public Node(int key, int val) {
            this.key = key; this.val = val;
        }
    }
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.put(key,value);
 * int param_2 = obj.get(key);
 * obj.remove(key);
 */
```

### [415] Add Strings

> sum of big integers

```java
class Solution {
    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, carry = 0;
        StringBuilder sb = new StringBuilder();
        while (i >= 0 || j >= 0 || carry > 0) {
            int sum = carry;
            sum += (i >= 0 ? num1.charAt(i--) - '0' : 0);
            sum += (j >= 0 ? num2.charAt(j--) - '0' : 0);
            sb.append(sum % 10);
            carry = sum / 10;
        }
        return sb.reverse().toString();
    }
}
```

> what if we want to compute the product?

### [43] Multiply Strings

> pos i \* pos j -> pos[i+j+1]pos[i+j], math

```
for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j, p2 = i + j + 1;
                int sum = mul + pos[p2];
                pos[p1] += sum / 10;
                pos[p2] = sum % 10;
            }
        }
```

### [739] Daily Temperatures

> Monotonous Stack

```java
class Solution {
    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < T.length; i++) {
            while (!stack.isEmpty() && T[i] > T[stack.peek()]) {
                int j = stack.pop();
                res[j] = i - j;
            }
            stack.push(i);
        }
        while(!stack.isEmpty()) res[stack.pop()] = 0;
        return res;
    }
}
```

> or you can use an array as a stack, which is faster

```java
class Solution {
    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        int[] stack = new int[T.length];
        int top = -1;
        for (int i = 0; i < T.length; i++) {
            while (top > -1 && T[i] > T[stack[top]]) {
                int j = stack[top--];
                res[j] = i - j;
            }
            stack[++top] = i;
        }
        while (top > -1) res[stack[top--]] = 0;
        return res;
    }
}
```

### [143] Reorder List

1. split the list into 2 lists in the middle
2. reverse the second list
3. merge 2 lists into one which is the final result

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return;
        ListNode slow = head, fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next; fast = fast.next.next;
        } // 1-2-3 slow->2 ; 1-2-3-4 slow->2 ;
        ListNode l2 = slow.next;
        slow.next = null;
        ListNode pre = null;
        while (l2 != null) {
            ListNode next = l2.next;
            l2.next = pre;
            pre = l2;
            l2 = next;
        }
        // merge head and rev l2
        ListNode l1 = head;
        l2 = pre;
        while (l1 != null && l2 != null) {
            ListNode next = l1.next;
            l1.next = l2;
            l2 = l2.next;
            l1.next.next = next;
            l1 = next;
        }
    }
}
```

> alternatively, we can use a stack for the second half

### [438] Find All Anagras in a String

> sliding window, [i, j] of fixed length p.length(), when j >= p.length() -1 we start to move i

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int[] arr = new int[26];
        for (char ch : p.toCharArray()) arr[ch - 'a']++;
        int i = 0, cnt = p.length();
        for (int j = 0; j < s.length(); j++) {
            if (--arr[s.charAt(j) - 'a'] >= 0) cnt--;
            if (cnt == 0) res.add(i);
            if (j >= p.length() - 1)
                if (++arr[s.charAt(i++) - 'a'] > 0) cnt++;
        }
        return res;
    }
}
```

> we can combine the two if statements into one

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int[] arr = new int[26];
        for (char ch : p.toCharArray()) arr[ch - 'a']++;
        int i = 0, cnt = p.length();
        for (int j = 0; j < s.length(); j++) {
            if (--arr[s.charAt(j) - 'a'] >= 0) cnt--;
            if (cnt == 0) res.add(i);
            if (j >= p.length() - 1 && ++arr[s.charAt(i++) - 'a'] > 0) cnt++;
        }
        return res;
    }
}
```