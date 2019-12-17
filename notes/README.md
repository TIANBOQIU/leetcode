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