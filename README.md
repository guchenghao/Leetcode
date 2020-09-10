## Leetcode
You can find the answers of some problems in leetcode, you can search the name of problem in leetcode and then the solution code would be found

## leetcode link
If you are intersted in my leetcode profile, you may go to the https://leetcode.com/guchenghao/

```

from math import log
from random import sample
from itertools import combinations, permutations
import random
import itertools
import collections
from fractions import Fraction
from collections import Counter
import operator
import re
from functools import reduce
from collections import defaultdict
import copy
import math
import numpy as np


# %%
# * Remove Element
def removeElement(nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    j = 0
    for i in range(len(nums)):
        if nums[i] == val:
            continue
        else:
            nums[j] = nums[i]
            j = j + 1

    return j


# %%
# * Search Insert Position
def searchInsert(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if target in nums:
        return nums.index(target)
    else:
        for i in range(len(nums)):
            if target < nums[i]:
                return i
            if target > nums[i]:
                if i == len(nums) - 1:
                    return i + 1
                continue


# %%
# * Missing Number
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()

    for i in range(len(nums)):
        if nums[0] != 0:
            return 0
        if nums[-1] != len(nums):
            return len(nums)
        if nums[i+1] - nums[i] == 1:
            continue
        else:
            return nums[i] + 1


# %%
# * Remove Duplicates from Sorted List
def deleteDuplicates(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    if head is None or head.next is None:
        return head
    seen = [head.val]
    prev_node = dum = head
    node = head.next
    while node is not None:
        if node.val in seen:
                # delete it
            prev_node.next = node.next
            node = node.next
        else:
            seen.append(node.val)
            prev_node = node
            node = node.next

    return dum


# %%
# * Length of Last Word
def lengthOfLastWord(self, s):
    """
    :type s: str
    :rtype: int
    """
    strArr = s.strip(' ').split(' ')
    return len(strArr[len(strArr) - 1])


# %%
# * Path Sum
# * 使用了递归的方式,遍历树
def hasPathSum(self, root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if root is None:
        return False

    if sum - root.val == 0 and root.left is None and root.right is None:
        return True
    else:
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


# %%
# * Contains Duplicate
def containsDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    nums.sort()
    if len(nums) == 1:
        return False
    for i in range(len(nums)):
        if i == len(nums) - 1:
            break
        if nums[i] == nums[i + 1]:
            return True
        else:
            continue
    return False


# %%
# * Contains Duplicate II
def containsNearbyDuplicate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    dic = {}
    for i, v in enumerate(nums):
        if v in dic and i - dic[v] <= k:
            return True
        dic[v] = i
    return False


# %%
# * Best Time to Buy and Sell Stock
# def maxProfit(self, prices):  # ! 超时
#     """
#     :type prices: List[int]
#     :rtype: int
#     """
#     result = []
#     for i in range(len(prices)):
#         if i == len(prices) - 1:
#             break
#         num_1 = prices[i + 1:]

#         if max(num_1) - prices[i] <= 0:
#             continue
#         else:
#             result.append(max(num_1) - prices[i])

#     if result == []:
#         return 0
#     else:
#         return max(result)
def maxProfit(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    n = len(prices)
    if n <= 1:
        return 0
    max_profit = 0
    low_price = prices[0]
    for i in range(1, n):
        low_price = min(low_price, prices[i])
        max_profit = max(max_profit, prices[i]-low_price)
    return max_profit


# %%
# * Majority Element


def majorityElement(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    num_dict = {}
    n = len(nums)
    if n == 1:
        return nums[0]
    for idn, v in enumerate(nums):
        if v in num_dict:
            num_dict[v] = num_dict[v] + 1
        else:
            num_dict[v] = 1
    result = max(num_dict.values())
    if result >= math.floor(n / 2):
        return max(num_dict, key=num_dict.get)


# %%
# * Find All Numbers Disappeared in an Array
def findDisappearedNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return list(set(range(1, len(nums)+1)) - set(nums))


# %%
# * Reshape the Matrix


def matrixReshape(self, nums, r, c):
    """
    :type nums: List[List[int]]
    :type r: int
    :type c: int
    :rtype: List[List[int]]
    """
    try:
        return np.reshape(nums, (r, c)).tolist()
    except:
        return nums


# %%
# * Largest Number At Least Twice of Others


def dominantIndex(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    copy_nums = copy.deepcopy(nums)
    nums.sort()
    maximum = nums[-1]
    for i in range(len(nums) - 1):
        if nums[i] == maximum:
            return -1
        else:
            if maximum >= nums[i] * 2:
                continue
            else:
                return -1
    return copy_nums.index(maximum)


# %%
# * Move Zeroes
def moveZeroes(self, nums):  # ! 这个方法慢
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    zero_num = nums.count(0)
    for i in range(zero_num):
        nums.remove(0)
        nums.append(0)


def moveZeroes(self, nums):  # ! 这个方法快100ms
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    length = len(nums)
    if length <= 1:
        return
    zero_count = 0
    i = 0
    while i < length:
        if nums[i] == 0:
            zero_count += 1
        else:
            nums[i - zero_count] = nums[i]
        i += 1
    j = 1
    while j <= zero_count:
        nums[length - j] = 0
        j += 1


# %%
# * Two Sum II - Input array is sorted


def twoSum(self, numbers, target):  # ! 方法可行，但超时了
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    for idn, v in enumerate(numbers):
        number_1 = np.array(numbers[idn + 1:])
        result = number_1 + v
        p = (result == target)

        if 1 in p:
            p = list(p)
            idn_2 = p.index(True)
            return [idn + 1, idn_2 + idn + 2]


def twoSum(self, numbers, target):  # ! 这个方法可行
    """
    :type numbers: List[int]
    :type target: int
    :rtype: List[int]
    """
    numbers_dict = {}
    for idn, v in enumerate(numbers):
        if target - v in numbers_dict:
            return [numbers_dict[target - v] + 1, idn + 1]
        numbers_dict[v] = idn


# %%
# * Max Consecutive Ones


def findMaxConsecutiveOnes(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums_dict = defaultdict(int)
    i = 0
    for num in nums:
        if num == 1:
            nums_dict[i] = nums_dict[i] + 1
        else:
            i = i + 1

    if nums_dict.values():
        return max(nums_dict.values())
    else:
        return 0


# %%
# * Maximum Product of Three Numbers


def maximumProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()

    return max(nums[1] * nums[0] * nums[-1], reduce(lambda x, y: x * y, nums[-3:]))


# %%
# * Longest Continuous Increasing Subsequence


def findLengthOfLCIS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums_dict = defaultdict(int)
    count = 1
    i = 0
    if nums == []:
        return 0
    for idn in range(len(nums) - 1):
        if nums[idn + 1] > nums[idn]:
            count = count + 1
            nums_dict[i] = count
        else:
            i = i + 1
            count = 1

    if nums_dict.values():
        return max(nums_dict.values())
    else:
        return 1


# %%
# * Find Pivot Index


def pivotIndex(self, nums):  # ! 超时了
    """
    :type nums: List[int]
    :rtype: int
    """
    for idn, num in enumerate(nums):
        if idn == 0:
            if reduce(lambda x, y: x + y, nums[idn + 1:]) == 0:
                return idn
            else:
                continue
        if idn == len(nums) - 1:
            if reduce(lambda x, y: x + y, nums[:idn]) == 0:
                return idn
            else:
                break
        if reduce(lambda x, y: x + y, nums[idn + 1:]) == reduce(lambda x, y: x + y, nums[:idn]):
            return idn

    return -1


def pivotIndex(self, nums):  # ! 143ms
    """
    :type nums: List[int]
    :rtype: int
    """
    left = 0
    right = sum(nums)
    for idn, num in enumerate(nums):
        right -= num
        if right == left:
            return idn
        left += num
    return -1


# %%
# * Plus One
def plusOne(self, digits):  # ! 68ms
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    num = 0
    for item in digits:
        num = num * 10 + item

    return [int(i) for i in str(num+1)]


# %%
# * Pascal's Triangle
def generate(self, numRows):  # ! 最直接的方式，双重循环, O(n^2)
    """
    :type numRows: int
    :rtype: List[List[int]]
    """
    if numRows == 0:
        return []
    result = []
    temp = [0, 1]

    for i in range(numRows):
        row = []
        for j in range(len(temp) - 1):
            row.append(temp[j] + temp[j+1])
        result.append(row)
        temp = row[:]
        temp.insert(0, 0)
        temp.append(0)

    return result


def generate(self, numRows):
    """
    :type numRows: int
    :rtype: List[List[int]]
    :example:
              1 3 3 1 0
           +  0 1 3 3 1
           =  1 4 6 4 1
    """
    if numRows == 0:
        return []

    res = [[1]]
    # ! "+" 在这里表示数组的拼接
    for _ in range(1, numRows):
        res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
    return res


# %%
# * Pascal's Triangle II
def getRow(self, rowIndex):
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [[1]]

    for _ in range(1, rowIndex + 1):
        res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]

    return res[rowIndex]


def getRow(self, rowIndex):  # ! 改良版,更快, 63ms
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [1]

    for _ in range(1, rowIndex + 1):
        res = list(map(lambda x, y: x+y, res + [0], [0] + res))

    return res


def getRow(self, rowIndex):  # ! 更快 55ms
    """
    :type rowIndex: int
    :rtype: List[int]
    """

    res = [1]
    for _ in range(1, rowIndex + 1):
        res = [x + y for x, y in zip([0]+res, res+[0])]

    return res


# %%
# * Average Subarray I
def findMaxAverage(self, nums, k):  # ! 超时
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    maxSum = -10001
    curSum = 0
    for i in range(len(nums) - k + 1):
        curSum = sum(nums[i:i + k])
        maxSum = max(maxSum, curSum)
    return float(maxSum / k)


def findMaxAverage(self, nums, k):  # ! 212ms
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    sums = [0] + list(itertools.accumulate(nums))
    # ! sub: sums[k:] 和 sums对应位置相减，不对应的位置舍弃
    return max(map(operator.sub, sums[k:], sums)) / k


def findMaxAverage(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    begin = 0
    end = 0
    n = len(nums)
    kavg = 0
    if k < len(nums):
        ksum = sum(nums[0:k])  # sum of first k items
        kavg = ksum / float(k)  # avg of first k items
        beg = 0  # beginning of avg window
        x = k  # end of avg window

        while x < n:  # when end falls beyond n
            ksum = ksum - nums[beg] + nums[x]      # get ksum
            # set kavg to maximum of kavg and new kavg
            kavg = max(kavg, (ksum / float(k)))
            beg += 1                               # begining of avg window increases
            x += 1                               # end of avg window increases
    else:
        # kavg if items are lesser than k
        kavg = sum(nums) / float(len(nums))
    return kavg


# %%
# * Can Place Flowers
def canPlaceFlowers(self, flowerbed, n):  # ! 118ms
    """
    :type flowerbed: List[int]
    :type n: int
    :rtype: bool
    """
    count = 0
    w = len(flowerbed) - 1
    for idn, v in enumerate(flowerbed):
        if count >= n:
            return True
        if v == 0 and idn == 0:
            if idn == w:
                count += 1
                break
            if flowerbed[idn + 1] != 1:
                count += 1
                flowerbed[idn] = 1
                continue
            continue
        if v == 0 and idn == w:
            if flowerbed[idn - 1] != 1:
                count += 1
                break
            break

        if v == 0 and flowerbed[idn + 1] != 1 and flowerbed[idn - 1] != 1:
            flowerbed[idn] = 1
            count += 1

    return count >= n


# %%
# * Third Maximum Number
def thirdMax(self, nums):  # ! 64ms
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    num_set = list(set(nums))
    num_set.sort()
    if len(num_set) < 3:
        return num_set[-1]

    return num_set[-3]


# %%
# * Rotate Array
def rotate(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    n = len(nums)
    k = k % n
    nums[:] = nums[n-k:] + nums[:n-k]


# %%
# * Shortest Unsorted Continuous Subarray
def findUnsortedSubarray(self, nums):  # ! 104ms
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 列表推导式
    res = [i for (i, (a, b)) in enumerate(
        zip(nums, sorted(nums))) if a != b]
    return 0 if not res else res[-1]-res[0]+1


# %%
# * Find All Duplicates in an Array
def findDuplicates(self, nums):  # ! 这种方式运行时间 322ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    nums.sort()
    result = []
    for i in range(len(nums) - 1):
        if nums[i] == nums[i + 1]:
            result.append(nums[i])

    return result


def findDuplicates(self, nums):  # ! 这种方式运行时间长 359ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    nums.sort()
    result = []
    i = 0
    while i < len(nums) - 1:
        if nums[i] == nums[i + 1]:
            result.append(nums[i])
            i = i + 2
        else:
            i = i + 1

    return result


# %%
# * Teemo Attacking
def findPoisonedDuration(self, timeSeries, duration):  # ! 138ms
    """
    :type timeSeries: List[int]
    :type duration: int
    :rtype: int
    """
    result = duration
    if timeSeries == []:
        return 0
    for i in range(len(timeSeries) - 1):
        if timeSeries[i + 1] > (timeSeries[i] + duration):
            result += duration
            continue
        else:
            result += (timeSeries[i + 1] - timeSeries[i])

    return result


# %%
# * Product of Array Except Self
def productExceptSelf(self, nums):  # ! 167ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    p = 1
    n = len(nums)
    output = []
    # ! [1,1,2,6]
    for i in range(0, n):
        output.append(p)
        p = p * nums[i]
    p = 1
    # ! [24,12,4,1]
    for i in range(n-1, -1, -1):
        output[i] = output[i] * p
        p = p * nums[i]
    return output  # ! [24,12,8,6]


# %%
# * Subsets
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = [[]]
    for num in nums:
        res += [item+[num] for item in res]
    return res


# %%
def numJewelsInStones(self, J, S):
    """
    :type J: str
    :type S: str
    :rtype: int
    """
    return sum(map(J.count, S))  # ! map函数的用法
    # return sum(s in J for s in S)


# %%
# * Keyboard Row


def findWords(self, words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    return list(filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words))


# %%
# * Distribute Candies
def distributeCandies(self, candies):
    """
    :type candies: List[int]
    :rtype: int
    """
    number_sister = len(candies) / 2

    candies_set = set(candies)

    if len(candies_set) <= number_sister:
        return len(candies_set)
    else:
        return int(number_sister)


# %%
# * Island Perimeter
def islandPerimeter(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    result = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] is 1:
                result += 4
                if j > 0 and grid[i][j-1] is 1:
                    result -= 2
                if i > 0 and grid[i-1][j] is 1:
                    result -= 2
    return result


# %%
# * Single Number
def singleNumber(self, nums):  # ! 93ms
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    i = 0
    if len(nums) == 1:
        return nums[0]
    while i <= len(nums) - 1:
        if i == len(nums) - 1:
            return nums[i]
        if nums[i] == nums[i + 1]:
            i = i + 2
            continue
        else:
            return nums[i]


def singleNumber(self, nums):  # ! 68ms
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 使用异或运算，比较巧妙
    res = 0
    for num in nums:
        res ^= num
    return res


# %%
# * Find the Difference
def findTheDifference(self, s, t):  # ! 118ms 遍历字符串
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    return [item for item in t if item not in s or s.count(item) != t.count(item)][0]


def findTheDifference(self, s, t):  # ! 68ms 采用异或
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    return chr(reduce(operator.xor, map(ord, s + t)))


def findTheDifference(self, s, t):
    # ! 分别两个字符串数组的所有字符的AscII码值总和，再相减
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    sum1 = sum(map(ord, [c for c in s]))
    sum2 = sum(map(ord, [c for c in t]))
    return chr(sum2-sum1)


# %%
# * First Unique Character in a String
# ! 如果使用遍历字符串的方式，会超时
def firstUniqChar(self, s):  # ! 90ms
    """
    :type s: str
    :rtype: int
    """
    # ! 只需要遍历26次
    letters = 'abcdefghijklmnopqrstuvwxyz'
    index = [s.index(l) for l in letters if s.count(l) == 1]
    return min(index) if len(index) > 0 else -1


# %%
# * Minimum Index Sum of Two Lists
def findRestaurant(self, list1, list2):
    """
    :type list1: List[str]
    :type list2: List[str]
    :rtype: List[str]
    """
    # ! 将 list转化为字典
    H = {string: idx for (idx, string) in enumerate(list1)}
    m, items = len(list2) + len(list1), []
    for i, string in enumerate(list2):
        if string in H:
            if i + H[string] < m:
                m = i + H[string]
                items = [string]  # ! 替换原本的数组
            elif i + H[string] == m:  # ! 判断 Sum of indexs 相等的情况
                items.append(string)  # ! 把元素添加进数组
    return items


# %%
# * Reverse String
def reverseString(self, s):
    """
    :type s: str
    :rtype: str
    """
    s_Arr = [item for item in s]
    s_Arr.reverse()
    return "".join(s_Arr)


# %%
# * Valid Anagram


def isAnagram(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    s_dict = Counter(s)
    t_dict = Counter(t)

    if s_dict - t_dict == {} and t_dict - s_dict == {}:
        return True
    else:
        return False


# %%
# * Intersection of Two Arrays
def intersection(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    nums1_set = set(nums1)
    nums2_set = set(nums2)

    return list(nums1_set & nums2_set)


# %%
# * Intersection of Two Arrays II


def intersect(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    c1 = Counter(nums1)
    c2 = Counter(nums2)
    return list((c1 & c2).elements())


# %%
# * Array Partition I
def arrayPairSum(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    return sum(nums[::2])


# %%
# * Set Mismatch
def findErrorNums(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]


# %%
# * Happy Number
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    mem = set()
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n in mem:
            return False
        else:
            mem.add(n)
    else:
        return True


# %%
# * Find All Anagrams in a String


def findAnagrams(self, s, p):  # ! 超时
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    p_dict = Counter(p)
    result = []
    # ! 因为每次循环都要调用Counter函数，所以总体时间会超时
    for i in range(len(s) - len(p) + 1):
        s_dict = Counter(s[i:i + len(p)])
        if s_dict - p_dict == {}:
            result.append(i)

    return result


def findAnagrams(self, s, p):  # ! 改良版 377ms
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    res = []
    pCounter = Counter(p)
    sCounter = Counter(s[:len(p)-1])
    for i in range(len(p)-1, len(s)):
        sCounter[s[i]] += 1   # include a new char in the window
        # This step is O(1), since there are at most 26 English letters
        if sCounter == pCounter:
            res.append(i-len(p)+1)   # append the starting index
        # decrease the count of oldest char in the window
        sCounter[s[i-len(p)+1]] -= 1
        if sCounter[s[i-len(p)+1]] == 0:
            del sCounter[s[i-len(p)+1]]   # remove the count if it is 0
    return res


# %%
# * Word Pattern
# ! 这道题目和Isomorphic Strings属于同类型题目
def wordPattern(self, pattern, str):
    """
    :type pattern: str
    :type str: str
    :rtype: bool
    """
    s = pattern
    t = str.split()
    return len(set(zip(s, t))) == len(set(s)) == len(set(t)) and len(s) == len(t)


# %%
# * Isomorphic Strings
def isIsomorphic(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    return len(set(zip(s, t))) == len(set(s)) == len(set(t))


# %%
# * Count Primes
def countPrimes(self, n):  # ! 厄拉多筛选法的变种
    """
    :type n: int
    :rtype: int
    """
    if n < 3:
        return 0
    primes = [True] * n
    primes[0] = primes[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if primes[i]:
            primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
    return sum(primes)


def countPrimes(self, n):  # ! 厄拉多筛选法 超时
    """
    :type n: int
    :rtype: int
    """
    if n < 3:
        return 0
    nums_list = list(range(1, n))
    nums_list[0] = 0
    for i in range(2, n):
        if nums_list[i-1] != 0:
            for j in range(i*2, n, i):
                nums_list[j-1] = 0
    result = [x for x in nums_list if x != 0]
    return len(result)


# %%
# * Delete Node in a Linked List
def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    # ! 从链表删除值为val的结点
    node.val = node.next.val
    node.next = node.next.next


# %%
# * Reverse Linked List
def reverseList(self, head):
    # ! 在遍历链表的时候，修改指针所指向的方向
    prev = None
    while head:
        curr = head
        curr.next = prev  # ! 调转指针方向
        prev = curr
        head = head.next
    return prev


# %%
# * Merge Two Sorted Lists
# ! 这种方法使用额外的空间，因为该程序创建了一个新的链表
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    dummy = cur = ListNode(0)
    # ! 如果两个链表中有一个链表被遍历完了，就退出循环，剩下的节点直接接入cur链表的表尾
    while l1 and l2:
        # ! l1和l2两个链表中，值小的那个接到cur链表尾部
        if l1.val < l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


# %%
# * Linked List Cycle
def hasCycle(self, head):  # ! Floyd判圈法(龟兔赛跑算法)
    """
    :type head: ListNode
    :rtype: bool
    """
    try:
        slow = head  # ! 龟
        fast = head.next  # ! 兔
        while slow is not fast:
            slow = slow.next
            fast = fast.next.next
        return True

    except AttributeError as e:
        return False


def hasCycle(self, head):  # ! 这个更快些
    """
    :type head: ListNode
    :rtype: bool
    """
    cur = head
    while cur:
        if cur.next == head:
            return True
        n = cur.next
        cur.next = head
        cur = n
    return False


# %%
# * Remove Linked List Elements
def removeElements(self, head, val):
    """
    :type head: ListNode
    :type val: int
    :rtype: ListNode
    """
    # ! 首先加个链表头dummy， 方便遍历第一个结点
    dummy = ListNode(-1)
    dummy.next = head
    cur = dummy

    while cur is not None and cur.next is not None:
        if cur.next.val == val:
            cur.next = cur.next.next  # ! 删除操作
        else:
            cur = cur.next  # ! 遍历下一个结点

    return dummy.next  # ! 返回原链表的首结点


# %%
# * Intersection of Two Linked Lists
def getIntersectionNode(self, headA, headB):
    """
    :type head1, head1: ListNode
    :rtype: ListNode
    """
    curA, curB = headA, headB
    lenA, lenB = 0, 0
    # ! 获取两个链表的长度
    while curA is not None:
        lenA += 1
        curA = curA.next
    while curB is not None:
        lenB += 1
        curB = curB.next
    # ! 让curA和curB指针复位
    curA, curB = headA, headB
    # ! 让指针在较长的链表中移两者的长度差
    if lenA > lenB:
        for i in range(lenA-lenB):
            curA = curA.next
    elif lenB > lenA:
        for i in range(lenB-lenA):
            curB = curB.next
    # !寻找两个链表中相同的结点
    while curB != curA:
        curB = curB.next
        curA = curA.next
    return curA


# %%
# * Self Dividing Numbers
def selfDividingNumbers(self, left, right):
    """
    :type left: int
    :type right: int
    :rtype: List[int]
    """
    def is_self_dividing(num): return '0' not in str(num) and all(
        [num % int(digit) == 0 for digit in str(num)])
    # ! 非常巧妙地利用了filter函数
    return list(filter(is_self_dividing, range(left, right + 1)))


# %%
# * Add Digits
def addDigits(self, num):
    """
    :type num: int
    :rtype: int
    """
    while len(str(num)) != 1:
        num = sum([int(item) for item in str(num)])

    else:
        return num


# %%
# * Excel Sheet Column Number
# ? A -> 1, B ->2
def titleToNumber(self, s):  # ! 这道题本质上是26进制转10进制
    """
    :type s: str
    :rtype: int
    """
    s = s[::-1]  # !将数组翻转，即reverse
    sum = 0
    for idn, char in enumerate(s):
        # ! ord函数用于计算字符的ASCII码值
        sum += (ord(char) - 65 + 1) * (26 ** idn)
    return sum


# %%
# * Power of Two
def isPowerOfTwo(self, n):
    """
    :type n: int
    :rtype: bool
    """

    while n % 2 == 0 and n is not 0:
        n = n / 2

    else:
        if n == 1:
            return True

    return False


# %%
# * Power of Three
def isPowerOfThree(self, n):  # ! 这道题和上一题一模一样，没什么意义
    """
    :type n: int
    :rtype: bool
    """
    while n % 3 == 0 and n is not 0:
        n = n / 3

    else:
        if n == 1:
            return True

    return False


# %%
# * Ugly Number
def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return False

    for item in (2, 3, 5):
        while num % item == 0:
            num = num / item

    return num == 1


# %%
# * Valid Perfect Square
def isPerfectSquare(self, num):  # ! 这个方法其实作弊了，这个方法和使用sqrt函数没什么区别
    """
    :type num: int
    :rtype: bool
    """
    if num == 0:
        return False
    return True if int(num**0.5)*int(num**0.5) == num else False


def isPerfectSquare(self, num):  # !这个代码有问题，一直通过不了
    r = num
    while r*r > num:
        r = (r + num/r) / 2
    return r*r == num


def isPerfectSquare(self, num):  # !这个方法是可行的，使用了牛顿法
    """
    :type num: int
    :rtype: bool
    """
    r = 1.0
    while abs(r*r - num) > 0.0001:
        r = (r + num/r) / 2
    return int(r) * int(r) == num


# %%
# * Sqrt(x)
def mySqrt(self, x):  # !只保留整数部分
    """
    :type x: int
    :rtype: int
    """
    r = 1.0
    # ! 利用牛顿下山法开根号
    while abs(r*r - x) > 0.0001:
        r = (r + x/r) / 2
    return int(r)


# %%
# * Excel Sheet Column Title
def convertToTitle(self, n):
    """
    :type n: int
    :rtype: str
    """
    # ! 这个字符数组的索引从0开始
    capitals = [chr(x) for x in range(ord('A'), ord('Z')+1)]
    result = []
    while n > 0:
        result.append(capitals[(n-1) % 26])
        n = (n-1) // 26  # ! // 取整除 - 返回商的整数部分
    result.reverse()
    return ''.join(result)


def convertToTitle(self, n):  # ! 改进版，减少了存储空间和运行时间
    """
    :type n: int
    :rtype: str
    """
    result = []
    while n > 0:
        result.append(chr((n-1) % 26 + 65))
        n = (n-1) // 26
    result.reverse()
    return ''.join(result)


# %%
# * Reverse Integer
def reverse(self, x):
    """
    :type x: int
    :rtype: int
    """
    result = []
    for item in str(x):
        result.insert(0, item)

    if str(x).find('-') > -1:
        result.insert(0, result.pop())

    n = int("".join(result))

    return n if n.bit_length() < 32 else 0


# %%
# * Add Binary
def addBinary(self, a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    # ! bin函数是返回一个二进制字符串
    return bin(int(a, 2)+int(b, 2))[2:]


# %%
# * Nth Digit
def findNthDigit(self, n):
    """
    :type n: int
    :rtype: int
    """
    # ! 这道题的核心思想是通过划分整数区间，在区间内找到相应数字
    if n < 0:
        return 0
    count = 9
    start = 1
    length = 1
    while n > (length * count):
        n -= length * count  # ! 确定n所在的区间,例如：n=150就在(10, 99)这个区间中
        length += 1
        start *= 10
        count *= 10
    start += (n - 1)/length  # ! 确定n落在该区间那个数字上
    return int((str(start)[(n - 1) % length]))


# %%
# * Sum of Square Numbers


def judgeSquareSum(self, c):
    """
    :type c: int
    :rtype: bool
    """
    # ! 这个方法中设置了两个指针 low 和 high
    # ! 类似于二分法
    low = 0
    high = int(math.sqrt(c))
    while low <= high:
        value = low*low+high*high
        if value < c:
            low += 1
        elif value > c:
            high -= 1
        else:
            return True
    return False


def judgeSquareSum(self, c):  # ! 这个方法思路比较简单
    """
    :type c: int
    :rtype: bool
    """
    half = int(math.sqrt(c))

    while half >= 0:  # ! 从sqrt(c)遍历到0
        if half ** 2 == c:
            return True
        another = int(math.sqrt(c - half ** 2))

        if another ** 2 == c - half ** 2:
            return True
        half -= 1

    return False


# %%
# * Perfect Number
def checkPerfectNumber(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 1:
        return False
    half = int(math.sqrt(num))
    result = []

    while half >= 1:
        another = num / half

        if num % half == 0:
            if half is not another:
                result.append(another)
                result.append(half)
            else:
                result.append(another)

        half -= 1

    return sum(result) - num == num


# %%
# * Factorial Trailing Zeroes
def trailingZeroes(self, n):  # ! 这个解法太妙了
    """
    :type n: int
    :rtype: int
    """

    zeroCnt = 0
    while n > 0:
        n = math.floor(n/5)
        zeroCnt += n

    return zeroCnt


def trailingZeroes(self, n):  # ! 这个方法是可行的，但是超时了
    """
    :type n: int
    :rtype: int
    """
    if n == 0:
        return 0
    result = reduce(lambda x, y: x * y, range(1, n + 1))
    count = 0
    result = Fraction(result, 1)
    while result % 10 == 0:
        count += 1
        result = result / 10

    return count


# %%
# * Arranging Coins
def arrangeCoins(self, n):  # ! 816ms, 比较慢
    """
    :type n: int
    :rtype: int
    """
    count = 0
    if n == 0:
        return 0
    while n > 0:
        count += 1
        n = n - count

    return count if n == 0 else count - 1


def arrangeCoins(self, n):  # ! 丧心病狂,利用等差数列的性质, 64ms
    """
    :type n: int
    :rtype: int
    """
    return int(((1+8*n) ** 0.5 - 1) / 2)


# %%
def strStr(self, haystack, needle):
    """
    :type haystack: str
    :type needle: str
    :rtype: int
    """

    if needle in haystack:
        return haystack.index(needle)
    else:
        return -1

    # ! return haystack.index(needle) if needle in haystack else -1


# %%
# * Palindrome Linked List


def isPalindrome(self, head):  # ! 这个方法使用了额外的空间 O(n)
    """
    :type head: ListNode
    :rtype: bool
    """
    # ! 生成一个双向队列 deque
    queue = collections.deque([])
    cur = head
    # ! 让所有的元素入队列
    while cur:
        queue.append(cur)
        cur = cur.next
    while len(queue) >= 2:
        # ! 每次把队列的一头一尾的元素进行比较
        if queue.popleft().val != queue.pop().val:
            return False
    return True


def isPalindrome(self, head):  # ! 是将链表的前半段进行翻转，然后和链表的后半段进行对比
    """
    :type head: ListNode
    :rtype: bool
    """
    rev = None
    slow = fast = head
    # ! 寻找链表的中间点
    while fast and fast.next:
        fast = fast.next.next
        # ! 翻转链表的前半部分
        rev, rev.next, slow = slow, rev, slow.next
    # ! 根据链表长度的奇偶，slow指针进行相应的变化
    if fast:
        slow = slow.next
    # ! 让链表的前半段和后半段进行比较
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev


# %%


def findPairs(self, nums, k):  # !这个逻辑没什么问题，但是代码太复杂了, 超时
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    count = 0
    if k < 0:
        return 0

    if k > 0:
        num_set = list(set(nums))
        for i in range(len(num_set)):
            if num_set[i] + k in num_set:
                count += num_set.count(num_set[i] + k)
        return count
    if k == 0:
        return sum(v > 1 for v in collections.Counter(nums).values())


def findPairs(self, nums, k):  # ! 这个就比较简洁了
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    # ! 分情况讨论
    res = 0
    c = collections.Counter(nums)  # ! 返回一个字典
    for i in c:  # ! 遍历字典的键值
        if k > 0 and i + k in c or k == 0 and c[i] > 1:
            res += 1
    return res


# %%
# * Valid Palindrome
def isPalindrome(self, s):  # ! 常规思路
    """
    :type s: str
    :rtype: bool
    """

    clean_s = [item.lower() for item in s if item.isalnum()]

    while len(clean_s) > 1:
        if clean_s.pop(0) == clean_s.pop():
            continue
        else:
            return False

    return True


def isPalindrome(self, s):  # !使用正则表达式, 操作有点骚
    """
    :type s: str
    :rtype: bool
    """
    # ! 替换s字符串中非数字和字母的所有字符
    s = re.sub(r'[^A-Za-z0-9]', '', s).lower()
    return s == s[::-1]


# %%
# * Reverse Words in a String III
def reverseWords(self, s):  # ! 常规方法
    """
    :type s: str
    :rtype: str
    """
    s_arr = s.split(" ")
    result = []
    for item in s_arr:
        result.append(item[::-1])

    return " ".join(result)


# %%
# * Detect Capital
def detectCapitalUse(self, word):
    """
    :type word: str
    :rtype: bool
    """
    # ! istitle函数是用来检测字符串中的单词首字母是否写
    # ! islower(isupper)函数是用来检测字符串中所有区分大小写的符号是否是小写(大写)
    return word.isupper() or word.islower() or word.istitle()


# %%
# * Ransom Note


def canConstruct(self, ransomNote, magazine):
    """
    :type ransomNote: str
    :type magazine: str
    :rtype: bool
    """
    m_dict = Counter(magazine)
    r_dict = Counter(ransomNote)

    for key, value in r_dict.items():
        if key in m_dict:
            if m_dict[key] >= value:
                continue
            else:
                return False
        else:
            return False

    return True


# %%
# * Reverse String II
def reverseStr(self, s, k):
    """
    :type s: str
    :type k: int
    :rtype: str
    """
    if len(s) == 1:
        return s

    if len(s) <= k:
        return s[::-1]

    if len(s) > k and len(s) < 2 * k:
        return s[k - 1::-1] + s[k:]

    if len(s) >= 2 * k:
        i = 0
        temp = []
        s_arr = [item for item in s]
        print(s_arr)
        while i < len(s_arr) - 1:
            temp = s_arr[i:i + k]
            temp.reverse()
            s_arr[i:i + k] = temp
            i += 2 * k

        return "".join(s_arr)


def reverseStr(self, s, k):
    """
    :type s: str
    :type k: int
    :rtype: str
    """
    i = 0
    while i*k < len(s):
        if i % 2 == 0:
            if (i+1)*k > len(s):
                s1 = s[i*k:]
                s = s[:i*k] + s1[::-1]
            else:
                s1 = s[i*k:(i+1)*k]
                s = s[:i*k] + s1[::-1] + s[(i+1)*k:]
        i += 1

    return s


# %%
# * Student Attendance Record I
def checkRecord(self, s):
    """
    :type s: str
    :rtype: bool
    """

    if s.count('A') >= 2:
        return False
    else:
        if 'LLL' not in s:
            return True
        else:
            return False

    # ! return False if s.count('A') >= 2 else True if s.count('LLL') == 0 else False  # 40ms
    # ! return s.count('A') <= 1 and 'LLL' not in s  # 36ms


# %%
# * String Compression


def compress(self, chars):
    """
    :type chars: List[str]
    :rtype: int
    """
    chars[:] = re.sub(r'(?<=(.))\1+', lambda m: str(1 +
                                                    len(m.group())), ''.join(chars))
    return len(chars)


# %%
# * Number of Segments in a String


def countSegments(self, s):
    """
    :type s: str
    :rtype: int
    """
    result = re.split(r'\s+', s)

    return len(result) - result.count('')
    # ! return len(s.split())


# %%
# * Judge Route Circle


def judgeCircle(self, moves):
    """
    :type moves: str
    :rtype: bool
    """
    moves_dict = Counter(moves)

    if moves_dict['U'] == moves_dict['D'] and moves_dict['R'] == moves_dict['L']:
        return True

    else:
        return False

    # ! return moves.count('L') == moves.count('R') and moves.count('U') == moves.count('D')


# %%
# * Valid Parentheses
# ! 主要使用栈stack
def isValid(self, s):  # ! 这个方法是比较巧妙，不过使用了额外的space,例如：map_dict、st
    """
    :type s: str
    :rtype: bool
    """
    map_dict = {')': '(', '}': '{', ']': '['}
    st = []
    for e in s:
        '''
          # ! 如果括号完全匹配的话，st应该为空，所以最后return not st
          # ! 只要st栈中还有剩余，就返回False
        '''
        if st and (e in map_dict and st[-1] == map_dict[e]):
            st.pop()  # ! 如果栈顶元素与当前元素匹配的话，弹出栈顶元素
        else:
            st.append(e)  # ! 如果栈顶元素与当前元素不匹配的话，压入栈中
    return not st


# %%
# * Repeated Substring Pattern
def repeatedSubstringPattern(self, s):  # ! 这种方法直接了当
    """
    :type s: str
    :rtype: bool
    """
    # ! 函数any的用法是：只要列表中有一个元素不为 False或None或''，就返回True
    # ! s[:i]从字符串头部开始取的原因是既然利用substring来reconstruct原本的字符串,必须满足字符串所有位置
    return any(s[:i] * (len(s) // i) == s for i in range(1, len(s) // 2 + 1) if len(s) % i == 0)


def repeatedSubstringPattern(self, s):  # ! 这个方法太巧妙了,一般很难能想出来
    """
    :type s: str
    :rtype: bool
    """
    # ! [1:-1]的意思是去除字符串中的第1和倒数第1位的元素
    if not s:
        return False  # ! 为空的话返回False

    return s in (2 * s)[1:-1]


# %%
# * Repeated String Match
def repeatedStringMatch(self, A, B):  # ! 这个方法思路清晰,但是太慢了
    """
    :type A: str
    :type B: str
    :rtype: int
    """
    C = ""
    for i in range(len(B)//len(A) + 3):
        if B in C:
            return i
        C += A

    return -1


def repeatedStringMatch(self, A, B):
    """
    :type A: str
    :type B: str
    :rtype: int
    """

    if B in A:
        return 1
    # ! 确认B中的所有元素都在A中，这样repeat才有意义
    if not (set(B).issubset(set(A))):
        return -1
    # ! 这一步是重点
    multiplier = -(-len(B) // len(A))  # ! 向上取整, 等同于math.ceil()
    getA = A
    getA *= multiplier
    if B in getA:
        return multiplier
    getA += A
    if B in getA:
        return multiplier + 1
    return -1


# %%
# * Hamming Distance
def hammingDistance(self, x, y):  # ! xor按位进行与操作，统计xor中'1'的个数
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    xor = x ^ y
    count = 0
    # ! _是一个没有具体意义的变量
    for _ in range(xor.bit_length()):
        count += xor & 1
        xor = xor >> 1
    return count


def hammingDistance(self, x, y):  # ! 利用count函数统计异或结果中'1'的数目
    """
    :type x: int
    :type y: int
    :rtype: int
    """

    return bin(x ^ y).count('1')


# %%
def findComplement(self, num):  # ! 这个方法相当于按位用1取扫描
    """
    :type num: int
    :rtype: int
    """
    i = 1

    while i <= num:
        i = i << 1

    return (i - 1) ^ num


def findComplement(self, num):
    """
    :type num: int
    :rtype: int
    """
    # ! bit_length函数不会获取一些没有意义的二进制位，比说：二进制数开始的许多没有意义的0
    return num ^ ((1 << num.bit_length())-1)


# %%
# *
def fizzBuzz(self, n):
    """
    :type n: int
    :rtype: List[str]
    """
    result = []

    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 != 0:
            result.append('Fizz')
            continue

        if i % 3 != 0 and i % 5 == 0:
            result.append('Buzz')
            continue

        if i % 3 == 0 and i % 5 == 0:
            result.append('FizzBuzz')
            continue

        result.append(str(i))

    return result
    # !下面这段代码是我代码的压缩版
    # ! return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]


# %%
# * Heaters
"""
遍历房屋houses，记当前房屋坐标为house：

    利用二分查找，分别找到不大于house的最大加热器坐标left，以及不小于house的最小加热器坐标right

    则当前房屋所需的最小加热器半径radius = min(house - left, right - house)

    利用radius更新最终答案ans
"""


def findRadius(self, houses, heaters):  # ! sort + 二分查找(binary search)
    """
    :type houses: List[int]
    :type heaters: List[int]
    :rtype: int
    """
    ans = 0
    heaters.sort()
    for house in houses:
        radius = 0x7FFFFFFF
        le = bisect.bisect_right(heaters, house)
        if le > 0:
            radius = min(radius, house - heaters[le - 1])
        ge = bisect.bisect_left(heaters, house)
        if ge < len(heaters):
            radius = min(radius, heaters[ge] - house)
        ans = max(ans, radius)
    return ans


# %%
# * Find Smallest Letter Greater Than Target
def nextGreatestLetter(self, letters, target):
    """
    :type letters: List[str]
    :type target: str
    :rtype: str
    """
    le = bisect.bisect_right(letters, target)

    if le == 0:
        return letters[0]

    if le > len(letters) - 1:
        return letters[0]

    return letters[le]


# %%
# * First Bad Version
# ! 折半查找
def firstBadVersion(self, n):
    """
    :type n: int
    :rtype: int
    """
    r = n-1
    le = 0
    while(le <= r):
        mid = le + (r-le)/2
        if not isBadVersion(mid):
            le = mid+1
        else:
            r = mid-1
    return le


# %%
# * Guess Number Higher or Lower
# ! 折半查找
def guessNumber(self, n):
    """
    :type n: int
    :rtype: int
    """
    low = 1
    high = n
    while True:
        mid = (low + high) / 2

        if guess(mid) == 1:
            low = mid + 1
        elif guess(mid) == 0:
            return mid
        else:
            high = mid - 1


# %%
# * Climbing Stairs
# ! 动态规划问题
def climbStairs(self, n):
    """
    :type n: int
    :rtype: int
    """
    if n <= 1:
        return 1
    res = []
    res.append(1)
    res.append(1)
    for i in range(2, n+1):
        res.append(res[-1]+res[-2])

    return res[-1]


# %%
# * Maximum Depth of Binary Tree
# ! 主要是使用递归的方式解题
def maxDepth(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0
    leftChildHeight = self.maxDepth(root.left)  # ! 左子树递归
    rightChildHeight = self.maxDepth(root.right)  # ! 右子树递归
    return max(leftChildHeight, rightChildHeight)+1


# %%
# * Binary Number with Alternating Bits
def hasAlternatingBits(self, n):
    """
    :type n: int
    :rtype: bool
    """
    bin_num = bin(n)[2:]

    for i in range(len(bin_num) - 1):
        if bin_num[i] is bin_num[i + 1]:
            return False

    return True


# %%
# * Invert Binary Tree
def invertTree(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if root:
        root.left, root.right = self.invertTree(
            root.right), self.invertTree(root.left)

    return root


# %%
# * Prime Number of Set Bits in Binary Representation
def countPrimeSetBits(self, L, R):
    """
    :type L: int
    :type R: int
    :rtype: int
    """
    def isprime(num):
        n = 2
        while n <= int(math.sqrt(num)):
            if i_count % n == 0:
                return False
            n += 1
        else:
            return True

    count = 0
    for i in range(L, R + 1):
        i_count = bin(i).count('1')

        if i_count is 1:
            continue

        if i_count <= 3:
            count += 1
        else:
            if isprime(i_count):
                count += 1

    return count


# %%
# * Reverse Bits
# ! 用到了zfill函数，字符串右对齐
def reverseBits(self, n):
    return int(bin(n)[2:].zfill(32)[::-1], 2)


# %%
# * Base 7
def convertToBase7(self, num):  # ! 短除法
    """
    :type num: int
    :rtype: str
    """
    if num == 0:
        return '0'
    n, res = abs(num), ''
    while n:
        res = str(n % 7) + res
        n //= 7
    return res if num > 0 else '-' + res


# %%
# * Sum of Left Leaves
def sumOfLeftLeaves(self, root):  # ! 利用深度优先搜索算法
    """
    :type root: TreeNode
    :rtype: int
    """
    self.sum = 0

    def depthTral(node):  # ! 深度优先遍历
        if node:
            if node.left is not None and node.left.right is None and node.left.left is None:
                self.sum += node.left.val

            depthTral(node.left)
            depthTral(node.right)

    depthTral(root)

    return self.sum


# %%
# * Same Tree
def isSameTree(self, p, q):
    """
    :type p: TreeNode
    :type q: TreeNode
    :rtype: bool
    """
    self.result = []
    self.result1 = []

    def depthTral(node, resultArr):  # ! 深度优先搜索
        if node:
            resultArr.append(node.val)
            if node.left:
                depthTral(node.left, resultArr)
            else:
                resultArr.append('')
            if node.right:
                depthTral(node.right, resultArr)
            else:
                resultArr.append('')

    depthTral(p, self.result)
    depthTral(q, self.result1)

    return self.result == self.result1


# %%
# * Assign Cookies
def findContentChildren(self, g, s):  # ! 使用贪心算法
    """
    :type g: List[int]
    :type s: List[int]
    :rtype: int
    """
    # ! 从小到大进行排序，可以消除原题带来的后效性
    g.sort()
    s.sort()
    # ! 双指针
    childi = 0
    cookiei = 0

    while cookiei < len(s) and childi < len(g):
        if s[cookiei] >= g[childi]:
            childi += 1
        cookiei += 1

    return childi


# %%
# * Range Sum Query - Immutable


class NumArray:  # ! 这个方法超时了

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums_arr = nums

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        result = reduce(lambda x, y: x + y, self.nums_arr[i:j + 1])

        return result


class NumArray:  # ! 使用itertools的accumulate方法

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.accu = [0] + list(itertools.accumulate(nums))

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.accu[j + 1] - self.accu[i]


# %%
def isPalindrome(self, x):  # ! 288ms, 时间有点长
    """
    :type x: int
    :rtype: bool
    """
    x = str(x)
    # if len(x) is 1: return True
    i = 0
    j = len(x) - 1
    while i <= len(x) / 2:
        if x[i] is not x[j]:
            return False
        i += 1
        j -= 1

    else:
        return True


# %%
def findRelativeRanks(self, nums):
    """
    :type nums: List[int]
    :rtype: List[str]
    """
    new_num = nums[:]
    num_dict = {}
    for i, v in enumerate(sorted(nums, reverse=True)):
        num_dict[v] = i + 1

    return ["Gold Medal" * (num_dict[item] is 1) or "Silver Medal" * (num_dict[item] is 2) or "Bronze Medal" * (num_dict[item] is 3) or str(num_dict[item]) for item in new_num]


# %%
# * Power of Four
def isPowerOfFour(self, num):
    """
    :type num: int
    :rtype: bool
    """
    while num % 4 == 0 and num is not 0:
        num = num / 4
    else:
        return True if int(num) is 1 else False

    # ! 下面的方法比较巧妙，利用了题设中的
    # ! return num != 0 and num &(num-1) == 0 and num & 1431655765== num


# %%
# * Minimum Depth of Binary Tree
def minDepth(self, root):  # ! 56ms 使用广度优先搜索
    """
    :type root: TreeNode
    :rtype: int
    """
    if root is None:
        return 0

    depth = 1

    queue = []
    queue.append(root)
    count = 0
    lastnum = 1

    while queue:
        node = queue.pop(0)
        print(node.val)

        if not node.left and not node.right:
            return depth
        lastnum -= 1

        if node.left:
            queue.append(node.left)
            count += 1
        if node.right:
            queue.append(node.right)
            count += 1

        if lastnum is 0:
            lastnum = count
            count = 0
            depth += 1


# %%
# * Min Stack
class MinStack:  # ! 768ms，比较慢

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return min(self.stack)


class MinStack:  # ! 这个方法快只需要64ms

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)
        if not self.stack2 or self.stack2[-1] >= x:
            self.stack2.append(x)

    def pop(self):
        tmp = self.stack1[-1]
        self.stack1.pop()
        if self.stack2 and tmp == self.stack2[-1]:
            self.stack2.pop()
        return tmp

    def top(self):
        return self.stack1[-1]

    def getMin(self):
        return self.stack2[-1]


# %%
# * Binary Watch
def readBinaryWatch(self, num):  # ! 44ms，进行双重循环，并统计1的数量，没什么太多的技巧性
    """
    :type num: int
    :rtype: List[str]
    """
    return ['%d:%02d' % (h, m)
            for h in range(12) for m in range(60)
            if (bin(h) + bin(m)).count('1') == num]


# %%
# * Binary Tree Paths
# ! 这道题的基本思路是采用深度优先搜索算法
def binaryTreePaths(self, root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    if not root.left and not root.right:
        return [str(root.val)]

    return [str(root.val) + '->' + i for i in self.binaryTreePaths(root.left)] + [str(root.val) + '->' + i for i in self.binaryTreePaths(root.right)]


def binaryTreePaths(self, root):  # ! 这个思路我想到了，但是实现的方式有点问题
    """
    :type root: TreeNode
    :rtype: List[str]
    """
    if not root:
        return []
    res = []
    self.dfs(root, "", res)
    return res

    def dfs(self, root, ls, res):
        if not root.left and not root.right:
            res.append(ls+str(root.val))
        if root.left:
            self.dfs(root.left, ls+str(root.val)+"->", res)
        if root.right:
            self.dfs(root.right, ls+str(root.val)+"->", res)


# %%
# * Second Minimum Node In a Binary Tree
# ! 使用深度优先搜索遍历每个结点
def findSecondMinimumValue(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    self.res = []
    if not root:
        return -1

    def dfs(node):
        if node:
            self.res.append(node.val)

            if node.left:
                dfs(node.left)

            if node.right:
                dfs(node.right)

    dfs(root)
    res_set = list(set(self.res))
    res_set.sort()

    return -1 if len(res_set) is 1 else res_set[1]


# %%
# * Implement Queue using Stacks
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        return self.stack.pop(0)

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        return self.stack[0]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return True if not self.stack else False


# %%
# * Convert a Number to Hexadecimal
def toHex(self, num):  # ! 这个方法是作弊的，我只是想测试一下时间，40ms
    """
    :type num: int
    :rtype: str
    """
    if num < 0:
        num += 2 ** 32

    return hex(num)[2:]


def toHex(self, num):  # ! 这个是常规方法， 37ms
    """
    :type num: int
    :rtype: str
    """
    if num is 0:
        return '0'
    if num < 0:
        num += 2 ** 32

    dict_hex = ['0', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    res = ''

    while num:

        res += dict_hex[num % 16]
        num = num // 16

    else:
        return res[::-1]


# %%
# * Implement Stack using Queues
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        return self.stack.pop()

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        return self.stack[-1]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        return True if not self.stack else False


# %%
# *  Two Sum IV - Input is a BST
def findTarget(self, root, k):  # ! 736ms，太慢了，使用深度优先搜索
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root:
        return False
    self.res = []

    def DSTFind(node):
        if node:
            if k - node.val in self.res:
                self.res.append('True')
                return

            self.res.append(node.val)

            if node.left:
                DSTFind(node.left)

            if node.right:
                DSTFind(node.right)

    DSTFind(root)

    return True if 'True' in self.res else False


def findTarget(self, root, k):  # ! 104ms，利用广度优先搜索
    """
    :type root: TreeNode
    :type k: int
    :rtype: bool
    """
    if not root:
        return False
    res = []
    bfs = [root]

    while bfs:
        node = bfs.pop(0)
        if k - node.val in res:
            return True

        res.append(node.val)

        if node.left:
            bfs.append(node.left)

        if node.right:
            bfs.append(node.right)

    return False


# %%
# * Construct the Rectangle
def constructRectangle(self, area):
    """
    :type area: int
    :rtype: List[int]
    """
    # ! int函数会自动向下取整
    mid = int(math.sqrt(area))
    while mid > 0:
        if area % mid == 0:
            # ! [L, W]
            return [int(area / mid), int(mid)]
        mid -= 1


# %%
# * Longest Common Prefix
def longestCommonPrefix(self, strs):  # ! 44ms
    """
    :type strs: List[str]
    :rtype: str
    """
    if not strs:
        return ""

    length = 0
    # ! 这个方法中主要是zip这方法用得十分巧妙
    # ! 每一个item就是字符串数组中每个字符串的字符组成一个tuple，例如：('','','')
    for item in list(zip(*strs)):

        if len(set(item)) > 1:
            return strs[0][:length]

        length += 1

    return strs[0][:length]


# %%
# * Average of Levels in Binary Tree
def averageOfLevels(self, root):  # ! 利用广度优先搜索，64ms
    """
    :type root: TreeNode
    :rtype: List[float]
    """
    if not root:
        return []
    bfs = [root]
    res = []
    tempres = []
    lastnum = 1
    count = 0

    while bfs:
        node = bfs.pop(0)
        tempres.append(node.val)
        lastnum -= 1

        if node.left:
            bfs.append(node.left)
            count += 1

        if node.right:
            bfs.append(node.right)
            count += 1

        if lastnum is 0:
            lastnum = count
            count = 0
            res.append(sum(tempres) / len(tempres))
            tempres = []

    return res


# %%
# * Longest Harmonious Subsequence
# ! 这道题其实就是再统计x和x + 1的总和最大的那组数据的长度


def findLHS(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    count = collections.Counter(nums)
    ans = 0
    for x in count:
        if x + 1 in count:
            ans = max(ans, count[x] + count[x+1])
    return ans


# %%
# * Top K Frequent Elements


def topKFrequent(self, nums, k):  # ! 使用 python的collections模块，O(nlogk)
    """
    :type nums: List[int]
    :type k: int
    :rtype: List[int]
    """

    return [item[0] for item in Counter(nums).most_common(k)]


# %%
# * Permutations


def permute(self, nums):  # ! 52ms，感觉作弊了
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    return list(itertools.permutations(nums))


# %%
# * Find the Duplicate Number


def findDuplicate(self, nums):  # ! 48ms
    """
    :type nums: List[int]
    :rtype: int
    """

    return Counter(nums).most_common(1)[0][0]


def findDuplicate(self, nums):  # ! 52ms
    """
    :type nums: List[int]
    :rtype: int
    """

    dict_nums = {}

    for idn, v in enumerate(nums):

        if v in dict_nums:
            return v

        dict_nums[v] = idn


# %%
# *
def singleNonDuplicate(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    left, right = 0, len(nums)-1
    while left < right:
        mid = left + (right-left)//2
        nei = mid+1 if mid % 2 == 0 else mid-1
        if nums[mid] == nums[nei]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


# %%
# * Single Number III


def singleNumber(self, nums):  # ! 56ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    result = []

    for item in Counter(nums).items():
        if item[1] == 1:
            result.append(item[0])

    return result


def singleNumber(self, nums):  # ! 48ms，这个方法快一些，思想是一样的
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    c = Counter(nums)
    L = []
    for x in nums:
        if c[x] == 1:
            L.append(x)
    return L


# %%
# * Binary Tree Inorder Traversal
def inorderTraversal(self, root):  # ! 40ms，中序遍历
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    self.result = []

    def inorderList(node):

        if node:

            inorderList(node.left)

            self.result.append(node.val)

            inorderList(node.right)

    inorderList(root)

    return self.result


# %%
# * Kth Largest Element in an Array
def findKthLargest(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """

    return sorted(nums, reverse=True)[k - 1]


# %%
# * Kth Smallest Element in a BST
# ! 二叉搜索树的性质: 左 < 中 < 右
def kthSmallest(self, root, k):  # ! 76ms
    """
    :type root: TreeNode
    :type k: int
    :rtype: int
    """
    res = []
    bfs = [root]

    while bfs:
        node = bfs.pop(0)
        res.append(node.val)

        if node.left:
            bfs.append(node.left)

        if node.right:
            bfs.append(node.right)

    return sorted(res)[k - 1]


def kthSmallest(self, root, k):  # ! 72ms，采用中序遍历
    """
    :type root: TreeNode
    :type k: int
    :rtype: int
    """
    def helper(node, count):
        if not node:
            return

        helper(node.left, count)
        count.append(node.val)
        helper(node.right, count)

    count = []
    helper(root, count)
    return count[k-1]


# %%
# * Find Largest Value in Each Tree Row
def largestValues(self, root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    bfs = [root]
    lastNum = 1
    res = []
    temp = []
    count = 0

    while bfs:
        node = bfs.pop(0)
        temp.append(node.val)
        lastNum -= 1

        if node.left:
            bfs.append(node.left)
            count += 1
        if node.right:
            bfs.append(node.right)
            count += 1
        if lastNum is 0:
            res.append(max(temp))
            lastNum = count
            count = 0
            temp = []

    return res


# %%
# * Shuffle an Array


class Solution:  # ! 520ms，使用python的random模块，其实也可以使用numpy random模块

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """

        return random.sample(self.nums, len(self.nums))


# %%
# * Find Bottom Left Tree Value
def findBottomLeftValue(self, root):  # ! 60ms，不是很快
    """
    :type root: TreeNode
    :rtype: int
    """
    bfs = [root]
    res = []
    temp = []
    lastNum = 1
    count = 0

    while bfs:
        node = bfs.pop(0)
        temp.append(node.val)
        lastNum -= 1

        if node.left:
            bfs.append(node.left)
            count += 1
        if node.right:
            bfs.append(node.right)
            count += 1
        if lastNum is 0:
            lastNum = count
            count = 0
            res.append(temp)
            temp = []

    return res[-1][0]


# %%
# * Sort Characters By Frequency


def frequencySort(self, s):
    """
    :type s: str
    :rtype: str
    """

    return "".join([char * times for (char, times) in Counter(s).most_common()])


# %%
# * Total Hamming Distance


def totalHammingDistance(self, nums):  # ! 超时了
    """
    :type nums: List[int]
    :rtype: int
    """
    res = 0

    for i in range(len(nums) - 1):
        temp = np.array(nums[i + 1:])
        res += sum([bin(item)[2:].count('1')
                    for item in list(nums[i] ^ temp)])

    return res


def totalHammingDistance(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 计算一项中有多少个(1,0)的组合
    return sum(b.count('0') * b.count('1') for b in zip(*map('{:032b}'.format, nums)))


# %%
# * Increasing Triplet Subsequence
def increasingTriplet(self, nums):  # ! 刚开始的时候理解错题意了
    """
    :type nums: List[int]
    :rtype: bool
    """
    first = second = float('inf')
    for n in nums:
        if n <= first:
            first = n
        elif n <= second:
            second = n
        else:
            return True
    return False


# %%
# * Combinations


def combine(self, n, k):  # ! 直接使用python的combinations函数，132ms
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    res = []
    digit = range(1, n + 1)

    for item in combinations(digit, k):
        res.append(list(item))
    return res


def combine(self, n, k):  # ! 这个是上面代码的简洁版，124ms
    """
    :type n: int
    :type k: int
    :rtype: List[List[int]]
    """
    return list(combinations(range(1, n + 1), k))


# %%
# * Binary Tree Preorder Traversal
def preorderTraversal(self, root):  # ! 前序遍历
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    self.res = []
    if not root:
        return []

    def perorder(node):

        if node:
            self.res.append(node.val)
            perorder(node.left)
            perorder(node.right)

    perorder(root)

    return self.res


# %%
# * Is Subsequence
def isSubsequence(self, s, t):  # ! 这个方法的思路比我之前的思路要清晰很多，虽然之前我想到了一些点，但是没有考虑清楚
    """
    :type s: str
    :type t: str
    :rtype: bool
    """
    for c in s:
        i = t.find(c)
        if i == -1:
            return False
        else:
            t = t[i+1:]
    return True


# %%
# * Combinations Sum III


def combinationSum3(self, k, n):  # ! 使用python的combination的API，36ms
    """
    :type k: int
    :type n: int
    :rtype: List[List[int]]
    """
    return [c for c in combinations(range(1, 10), k) if sum(c) == n]


# %%
# * Pow(x, n)
def myPow(self, x, n):  # ! 我看不出这道题的意义何在，题目比较智障
    """
    :type x: float
    :type n: int
    :rtype: float
    """
    return x ** n


# %%
# * Super Pow
# ! 我尝试了其他几种方法都超时了.....
def superPow(self, a, b):  # ! 直接使用pow和map函数
    """
    :type a: int
    :type b: List[int]
    :rtype: int
    """
    return pow(a, int("".join(map(str, b))), 1337)


# %%
# * Single Number II


def singleNumber(self, nums):  # ! 40ms
    """
    :type nums: List[int]
    :rtype: int
    """

    return Counter(nums).most_common()[-1][0]


# %%
# * Kth Smallest Element in a Sorted Matrix
def kthSmallest(self, matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """
    return sorted([element for row in matrix for element in row])[k-1]


# %%
# * Unique Morse Code Words
def uniqueMorseRepresentations(self, words):
    """
    :type words: List[str]
    :rtype: int
    """
    vacab = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
             "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    res = []

    for item in words:
        string = "".join([vacab[ord(s) - 97] for s in item])
        res.append(string)

    return len(set(res))


# %%
# * Multiply Strings
def multiply(self, num1, num2):  # ! 这虽然是作弊，但是效果不错 52ms
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    return str(int(num1) * int(num2))


# %%
# * Add Two Numbers II
def addTwoNumbers(self, l1, l2):  # ! 这个方法的思路非常的简单，所以运行时间比较长
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    c1, c2 = '', ''
    while l1:
        c1 += str(l1.val)
        l1 = l1.next
    while l2:
        c2 += str(l2.val)
        l2 = l2.next
    num = str(int(c1) + int(c2))
    dummy = ListNode(0)
    c = dummy
    for i in range(len(num)):
        c.next = ListNode(num[i])
        c = c.next
    return dummy.next


# %%
# * Map Sum Pairs
class MapSum:  # ! 使用了startswith函数，36ms

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dict_MS = {}

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: void
        """
        self.dict_MS[key] = val

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        res = 0
        for item in self.dict_MS.items():
            if item[0].startswith(prefix):
                res += item[1]
        return res


# %%
# * Number of Lines To Write String
def numberOfLines(self, widths, S):  # ! 这道题我理解错题意了
    """
    :type widths: List[int]
    :type S: str
    :rtype: List[int]
    """
    total = 0
    lines = 0
    for char in S:
        total += widths[ord(char) - ord("a")]
        # move to a new line
        if total > 100:
            total = widths[ord(char) - ord("a")]
            lines += 1

    return [lines+1, total]


# %%
# * Add Strings
def addStrings(self, num1, num2):  # ! 48ms，这个方法作弊了
    """
    :type num1: str
    :type num2: str
    :rtype: str
    """
    return str(int(num1) + int(num2))


# %%
# * Most Common Word


def mostCommonWord(self, paragraph, banned):
    """
    :type paragraph: str
    :type banned: List[str]
    :rtype: str
    """
    ban = set(banned)
    # ! 将paragraph中所有非字母的符号转化为空格
    words = re.sub(r'[^a-zA-Z]', ' ', paragraph).lower().split()
    # ! 除去banned数组中的值，再利用Counter获取最高频率的词
    return Counter(w for w in words if w not in ban).most_common(1)[0][0]


# %%
# * Bulb Switcher


def bulbSwitch(self, n):  # ! 这是一道头脑风暴题，找规律(笑哭)
    """
    :type n: int
    :rtype: int
    """
    return int(math.sqrt(n))


# %%
# * Goat Latin
def toGoatLatin(self, S):  # ! 36ms，这道题太蠢了，直接按照要求做
    """
    :type S: str
    :rtype: str
    """
    S = S.split()
    vowel_dict = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
    res = []

    for idn, v in enumerate(S):
        if v[0] in vowel_dict:
            res.append(v + 'ma' + 'a' * (idn + 1))
        else:
            res.append(v[1:] + v[0] + 'ma' + 'a' * (idn + 1))

    return " ".join(res)


# %%
# * Array Nesting
def arrayNesting(self, nums):  # ! 超时了
    """
    :type nums: List[int]
    :rtype: int
    """
    res = []
    temp = []
    # dict_nums = {k:v for k, v in enumerate(nums)}

    for item in nums:
        i = item
        count = 0
        while i <= len(nums) - 1:
            if i in temp:
                break
            else:
                temp.append(i)
                count += 1
                i = nums[i]

        res.append(count)

    return max(res)


def arrayNesting(self, nums):  # ! 通过设置访问数组来防止重复访问，80ms
    """
    :type nums: List[int]
    :rtype: int
    """
    ans, step, n = 0, 0, len(nums)
    seen = [False] * n
    for i in range(n):
        while not seen[i]:
            seen[i] = True
            i, step = nums[i], step + 1
        ans = max(ans, step)
        step = 0
    return ans


# %%
# * Rotate String
def rotateString(self, A, B):  # ! 44ms，python数组切片
    """
    :type A: str
    :type B: str
    :rtype: bool
    """
    if A == B and not A and not B:
        return True
    temp = ''
    for i in range(len(A) - 1):
        temp = A[i + 1:] + A[0:i + 1]
        if temp == B:
            return True

    return False


# %%
# * Find Mode in Binary Search Tree


def findMode(self, root):  # ! 76ms，广度优先搜索
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    BST = [root]
    res = []
    final = []

    while BST:
        node = BST.pop(0)
        res.append(node.val)

        if node.left:
            BST.append(node.left)

        if node.right:
            BST.append(node.right)

    if len(res) == 1:
        return res

    res = Counter(res).most_common()
    print(res)

    final.append(res[0][0])

    for i in range(len(res) - 1):
        if res[i + 1][1] == res[i][1]:
            final.append(res[i + 1][0])
            continue
        else:
            break

    return final


# %%
# * Binary Tree Level Order Traversal II
def levelOrderBottom(self, root):  # ! 44ms，广度优先搜索
    """
    :type root: TreeNode
    :rtype: List[List[int]]
    """
    if not root:
        return []
    BST = [root]
    res = []
    temp = []
    levelNodeNum = 1
    count = 0

    while BST:
        node = BST.pop(0)
        temp.append(node.val)
        levelNodeNum -= 1
        if node.left:
            count += 1
            BST.append(node.left)

        if node.right:
            count += 1
            BST.append(node.right)

        if levelNodeNum is 0:
            res.insert(0, temp)
            levelNodeNum = count
            count = 0
            temp = []

    return res


# %%
# * Sort Colors
def sortColors(self, nums):  # ! 36ms，我感觉这道题在搞事情
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    nums.sort()


def sortColors(self, nums):  # ! 使用collections中Counter接口，40ms，In-place
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    pre = cur = 0
    for item in Counter(nums).items():
        cur += item[1]
        nums[pre:cur] = [item[0]] * item[1]
        pre = cur


# %%
# * Median of Two Sorted Arrays
def findMedianSortedArrays(self, nums1, nums2):  # ! 104ms
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    new_num = sorted(nums1 + nums2)

    if len(new_num) % 2 is 0:
        return float(new_num[len(new_num) // 2] + new_num[(len(new_num) // 2) - 1]) / 2
    else:
        return new_num[len(new_num) // 2]


# %%
# * Merge Intervals
def merge(self, intervals):  # ! 这个方法既巧妙又简单，64ms
    """
    :type intervals: List[Interval]
    :rtype: List[Interval]
    """

    res = []
    # ! 先将intervals对象数组进行排序
    for item in sorted(intervals, key=lambda item: item.start):
        if res and item.start <= res[-1].end:
            res[-1].end = max(res[-1].end, item.end)
        else:
            res.append(item)

    return res


# %%
# * Insert Delete GetRandom O(1)


class RandomizedSet:  # ! 244ms，虽然未超时，但时间太长了

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.randomSet = set()

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.randomSet:
            self.randomSet.add(val)
            return True
        return False

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.randomSet:
            self.randomSet.remove(val)
            return True
        return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        # return choice(list(self.randomSet)) # ! 换成这个会快一些，204ms
        return sample(list(self.randomSet), 1)[0]


# %%
# * Container With Most Water
def maxArea(self, height):  # ! 72ms，这个方法很牛逼，贪心算法
    """
    :type height: List[int]
    :rtype: int
    """
    i, j = 0, len(height) - 1
    water = 0
    # ! 先考虑容器底边最长，再移动较短的容器壁，这样能得到局部最优解
    # ! 这个点比较难想到，不容易证明
    while i < j:
        water = max(water, (j - i) * min(height[i], height[j]))
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return water


# %%
# * Remove Duplicates from Sorted Array II


def removeDuplicates(self, nums):  # ! 这个方法太烂了，暴力求解，92ms，时间过长
    """
    :type nums: List[int]
    :rtype: int
    """

    for item in Counter(nums).items():
        if item[1] > 2:
            for i in range(item[1] - 2):
                nums.remove(item[0])
        else:
            continue


def removeDuplicates(self, nums):  # ! 72ms，这个方法太巧妙了
    """
    :type nums: List[int]
    :rtype: int
    """

    i = 0
    for n in nums:
        if i < 2 or n > nums[i-2]:
            nums[i] = n
            i += 1
    return i


# %%
# * Search in Rotated Sorted Array II
def search(self, nums, target):  # ! 44ms，这道题智障吗？
    """
    :type nums: List[int]
    :type target: int
    :rtype: bool
    """
    if target in nums:
        return True

    return False

    # return target in nums  # ! 一行代码搞定，下次可以把代码写得更加简洁一些


# %%
# * Search in Rotated Sorted Array
def search(self, nums, target):  # ! 40ms，人生苦短，我用python
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    try:
        return nums.index(target)
    except:
        return -1


# %%
# * Find Minimum in Rotated Sorted Array
def findMin(self, nums):  # ! 36ms，只想说python牛逼
    """
    :type nums: List[int]
    :rtype: int
    """
    return sorted(nums)[0]

    # return min(nums)


# %%
# * Find Minimum in Rotated Sorted Array II
def findMin(self, nums):  # ! 40ms，我无语了
    """
    :type nums: List[int]
    :rtype: int
    """
    return sorted(nums)[0]


# %%
# * Lexicographical Numbers
def lexicalOrder(self, n):  # ! 156ms，巧用map函数
    """
    :type n: int
    :rtype: List[int]
    """

    return list(map(int, sorted(map(str, range(1, n + 1)))))


# %%
# * Replace Words
def replaceWords(self, dict, sentence):  # ! 224ms，虽然AC了，时间太长了，而且使用了双重循环
    """
    :type dict: List[str]
    :type sentence: str
    :rtype: str
    """
    sentence_list = sentence.split()

    for i in range(len(sentence_list)):
        for item in dict:
            if sentence_list[i].startswith(item):
                sentence_list[i] = item

    return " ".join(sentence_list)


# %%
# * Flipping an Image
def flipAndInvertImage(self, A):  # ! 48ms
    """
    :type A: List[List[int]]
    :rtype: List[List[int]]
    """

    for item in A:
        item[:] = item[::-1]

        for i in range(len(item)):
            if item[i] == 0:
                item[i] = 1
            else:
                item[i] = 0

    return A


# %%
# * Majority Element II


def majorityElement(self, nums):  # ! 44ms
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    res = []

    for item in Counter(nums).items():
        if item[1] > len(nums) // 3:
            res.append(item[0])

    return res


# %%
# * Search for a Range
def searchRange(self, nums, target):  # ! 40ms
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    res = []
    dict_nums = {}
    if target not in nums:
        return [-1, -1]
    else:
        firstpoi = nums.index(target)
        if firstpoi == len(nums) - 1:
            return [firstpoi, firstpoi]
        for i in range(firstpoi, len(nums)):
            if nums[i + 1] == nums[i]:
                if i + 1 == len(nums) - 1:
                    return [firstpoi, i + 1]
                continue
            else:
                return [firstpoi, i]


# %%
def isRectangleOverlap(self, rec1, rec2):  # ! 这是一道数学题
    """
    :type rec1: List[int]
    :type rec2: List[int]
    :rtype: bool
    """

    return rec1[0] < rec2[2] and rec2[0] < rec1[2] and rec1[1] < rec2[3] and rec2[1] < rec1[3]


# %%
# * Number of Digit One
def countDigitOne(self, n):  # ! 超时了
    """
    :type n: int
    :rtype: int
    """
    counter = 0

    if n > 0:
        for i in range(1, n + 1):
            counter += str(i).count('1')

        return counter

    else:

        return 0


# %%
# * Divide Two Integers
def divide(self, dividend, divisor):  # ! 这道题垃圾，不解释
    """
    :type dividend: int
    :type divisor: int
    :rtype: int
    """
    positive = (dividend < 0) is (divisor < 0)
    dividend, divisor = abs(dividend), abs(divisor)
    res = 0
    while dividend >= divisor:
        temp, i = divisor, 1
        while dividend >= temp:
            dividend -= temp
            res += i
            i <<= 1
            temp <<= 1
    if not positive:
        res = -res
    return min(max(-2147483648, res), 2147483647)


# %%
# * Rotate Image


def rotate(self, matrix):  # ! 92ms，使用numpy的rot90函数
    """
    :type matrix: List[List[int]]
    :rtype: void Do not return anything, modify matrix in-place instead.
    """
    temp = np.array(matrix)
    # ! tolist方法是将array或matrix转化为list
    temp = np.rot90(temp, -1).tolist()

    for i in range(len(temp)):
        matrix[i] = temp[i]

    # matrix[:] = temp[:] # ! 等价于上面的循环


# %%
# * Complex Number Multiplication
def complexNumberMultiply(self, a, b):  # ! 纯数学题，没什么意义， 36ms
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    # ! 先从string中获取复数的系数
    ar, ai = a[:-1].split('+')
    br, bi = b[:-1].split('+')
    # ! 根据复数乘法法则计算结果
    nr = int(ar)*int(br) - int(ai)*int(bi)
    ni = int(ar)*int(bi) + int(ai)*int(br)
    return "{}+{}i".format(nr, ni)


# %%
# * Reverse Words in a String
def reverseWords(self, s):  # ! 这道题有点弱智了， 32ms
    """
    :type s: str
    :rtype: str
    """
    return " ".join(s.split()[::-1])


# %%
# * Valid Square
def validSquare(self, p1, p2, p3, p4):  # ! 验证正方形的要点是四个点两两相减，出现的差值只有两种(整数为前提条件)
    """
    :type p1: List[int]
    :type p2: List[int]
    :type p3: List[int]
    :type p4: List[int]
    :rtype: bool
    """
    d1 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    d2 = (p1[0] - p3[0])**2 + (p1[1] - p3[1])**2
    d3 = (p1[0] - p4[0])**2 + (p1[1] - p4[1])**2
    d4 = (p2[0] - p3[0])**2 + (p2[1] - p3[1])**2
    d5 = (p2[0] - p4[0])**2 + (p2[1] - p4[1])**2
    d6 = (p3[0] - p4[0])**2 + (p3[1] - p4[1])**2

    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0 or d5 == 0 or d6 == 0:
        return False

    if len(set([d1, d2, d3, d4, d5, d6])) == 2:
        return True

    return False


# %%
# * Compare Version Numbers
def compareVersion(self, version1, version2):  # ! 40ms，思路没问题，但是思考时陷入了死胡同
    """
    :type version1: str
    :type version2: str
    :rtype: int
    """
    v1 = version1.split(".")
    v2 = version2.split(".")
    # ! 将两个字符串的长度补全到相等，这点很重要
    while len(v1) < len(v2):
        v1.append("0")
    while len(v1) > len(v2):
        v2.append("0")

    for i in range(len(v1)):
        if int(v1[i]) < int(v2[i]):
            return -1
        elif int(v1[i]) > int(v2[i]):
            return 1
    return 0


# %%
# * Binary Gap
def binaryGap(self, N):
    """
    :type N: int
    :rtype: int
    """
    bin_n = bin(N)
    temp = []
    maximum = -float('Inf')

    for i in range(2, len(bin_n)):
        if bin_n[i] == '1':
            temp.append(i)

    if len(temp) == 1:
        return 0
    else:
        for i in range(len(temp) - 1):
            test = temp[i + 1] - temp[i]
            if maximum < test:
                maximum = test

        return maximum


# %%
# * Peak Index in a Mountain Array
def peakIndexInMountainArray(self, A):  # ! 时间略长了些
    """
    :type A: List[int]
    :rtype: int
    """
    return A.index(max(A))


# %%
# * N-ary Tree Level Order Traversal
def levelOrder(self, root):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    if not root:
        return []

    queue = [root]
    count = 0
    lastnum = 1
    temp = []
    res = []

    while queue:
        node = queue.pop(0)
        temp.append(node.val)
        lastnum -= 1

        if node.children:
            test = [child for child in node.children]
            queue += test
            count += len(test)

        if lastnum is 0:
            lastnum = count
            count = 0
            res.append(temp)
            temp = []
    return res


# %%
# * N-ary Tree Preorder Traversal
def preorder(self, root):
    """
    :type root: Node
    :rtype: List[int]
    """
    self.res = []
    if not root:
        return []

    def perorder_final(node):
        if node:
            self.res.append(node.val)

            if node.children:
                for child in node.children:
                    perorder_final(child)

    perorder_final(root)

    return self.res


# %%
# * N-ary Tree Postorder Traversal
def postorder(self, root):  # ! 后序遍历
    """
    :type root: Node
    :rtype: List[int]
    """
    self.res = []
    if not root:
        return []

    def postorder_final(node):
        if node:
            if node.children:
                for child in node.children:
                    postorder_final(child)
            self.res.append(node.val)

    postorder_final(root)

    return self.res


# %%
# * Keys and Rooms
def canVisitAllRooms(self, rooms):
    """
    :type rooms: List[List[int]]
    :rtype: bool
    """
    size = len(rooms)
    visited = set()
    q = []
    q.append(0)
    while q:
        room = q.pop()
        visited.add(room)
        for key in rooms[room]:
            if key not in visited:
                q.append(key)
    return len(visited) == size


# %%
# * Missing Number
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    if nums[0] is not 0:
        return 0
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] > 1:
            return nums[i] + 1

    return nums[-1] + 1


# %%
# * Calculate Entropy


class Solution(object):
    def calculateEntropy(self, input):
        """
        :type input: List[int]
        :rtype: float
        """

        input_dict = Counter(input)

        prob = [float(item) / len(input) for item in input_dict.values()]

        result = 0.0

        for item in prob:
            result -= item * log(item, 2)

        return result


# %%
# * Calculate Maximum Information Gain


class Solution(object):
    def calculateMaxInfoGain(self, petal_length, species):
        """
        :type petal_length: List[float]
        :type species: List[str]
        :rtype: float
        """
        def calculateEntropy(input):
            """
            :type input: List[int]
            :rtype: float
            """

            input_dict = Counter(input)
            prob = [float(item) / len(input) for item in input_dict.values()]
            result = 0.0

            for item in prob:
                result -= item * log(item, 2)

            return result

        if len(species) == 0:
            return 0.0

        tuple_test = zip(petal_length, species)
        tuple_test.sort(key=lambda x: x[0])
        print(tuple_test)

        species_sort = [item[1] for item in tuple_test]
        total_en = calculateEntropy(species_sort)
        total_size = len(petal_length)

        final_en = float('inf')

        for i in range(1, total_size):
            min_en = calculateEntropy(species_sort[:i]) * (float(len(species_sort[:i])) / total_size) + calculateEntropy(
                species_sort[i:]) * (float(len(species_sort[i:])) / total_size)
            if min_en < final_en:
                final_en = min_en

        return total_en - final_en

# %%
# * Search for a Range


def searchRange(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    res_indx = [-1, -1]

    for i in range(len(nums)):
        if nums[i] == target:
            res_indx[0] = i
            break

    for i in range(len(nums) - 1, -1, -1):
        if nums[i] == target:
            res_indx[1] = i
            break

    return res_indx


# %%
# * Find K Closest Elements
def findClosestElements(self, arr, k, x):
    """
    :type arr: List[int]
    :type k: int
    :type x: int
    :rtype: List[int]
    """
    tuple_test = zip([abs(item - x) for item in arr], arr)
    tuple_test.sort(key=lambda x: x[0])

    return sorted([item[1]for item in tuple_test[:k]])


# %%
# * Valid Perfect Square
def isPerfectSquare(self, num):
    # ! memory超出
    """
    :type num: int
    :rtype: bool
    """
    if num == 1:
        return True
    for i in range(2, num - 1):
        if num % i == 0 and num / i == i:
            return True

    return False


def isPerfectSquare(self, num):
    """
    :type num: int
    :rtype: bool
    """
    # ! binary search
    # ! This is template 1.
    # ! And template 1 is used to search for an element or condition which can be determined by accessing a single index in the array
    if num == 0:
        return False

    left, right = 0, num

    while left <= right:

        mid = (left + right) // 2

        if mid * mid == num:
            return True
        elif mid * mid > num:
            right = mid - 1
        else:
            left = mid + 1

    return False


# %%
# * Find Smallest Letter Greater Than Target
def nextGreatestLetter(self, letters, target):
    """
    :type letters: List[str]
    :type target: str
    :rtype: str
    """
    if ord(target) >= ord(letters[-1]):
        return letters[0]
    else:
        for i in range(len(letters)):
            if ord(target) < ord(letters[i]):
                return letters[i]


# %%
# * Find K-th Smallest Pair Distance


class Solution(object):
    def smallestDistancePair(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # ! 暴力求解超内存限制
        all_pairs = list(combinations(nums, 2))

        all_pairs.sort(key=lambda x: abs(x[0] - x[1]))

        return abs(all_pairs[k - 1][0] - all_pairs[k - 1][1])


# %%
# * Add and Search Word - Data structure design

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.word_dict = defaultdict(set)

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        self.word_dict[len(word)].add(word)

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        for w in list(self.word_dict[len(word)]):
            for item in zip(w, word):
                if item[1] not in [".", item[0]]:
                    break
            else:
                return True

        return False


# %%
# * Number of Islands
def numIslands(self, grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    # ! 广度优先搜索(bfs)，实际上是去遍历网格中为1的点，去判断其上下左右四个方向上相邻的点
    if not grid:
        return 0

    row, col = len(grid), len(grid[0])  # ! 获取长宽
    land = set([(i, j) for i in range(row)
                for j in range(col) if grid[i][j] == "1"])
    res = 0

    while land:
        res += 1

        q_bfs = [land.pop()]

        while q_bfs:

            i, j = q_bfs.pop(0)

            for item in [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]:
                if item in land:
                    land.remove(item)
                    q_bfs.append(item)

    return res


# %%
# * Decode String
def decodeString(self, s):
    """
    :type s: str
    :rtype: str
    """
    # ! 利用栈进行求解
    stack = []

    for item in s:

        if item != "]":
            stack.append(item)

        else:
            tmp_s = ""

            while stack:
                str_x = stack.pop()

                if str_x == "[":
                    n = ""

                    while stack and stack[-1].isdigit():
                        n = stack.pop() + n

                    stack.append(tmp_s * int(n))
                    break
                else:
                    tmp_s = str_x + tmp_s

    return "".join(stack)


# %%
# * 01 Matrix
def updateMatrix(self, matrix):
    """
    :type matrix: List[List[int]]
    :rtype: List[List[int]]
    """
    # ! 这道题以0为基准进行BFS
    if not matrix:
        return [[]]

    row, col = len(matrix), len(matrix[0])
    queue = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == 0:
                queue.append((i, j))

            else:
                matrix[i][j] = -1

    while queue:
        i, j = queue.pop()

        for cor in [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]:
            if 0 <= cor[0] < row and 0 <= cor[1] < col and matrix[cor[0]][cor[1]] < 0:
                matrix[cor[0]][cor[1]] = matrix[i][j] + 1
                queue.insert(0, cor)

    return matrix


# %%
# * Flood Fill
def floodFill(self, image, sr, sc, newColor):
    """
    :type image: List[List[int]]
    :type sr: int
    :type sc: int
    :type newColor: int
    :rtype: List[List[int]]
    """
    # ! 利用BFS的方法进行求解
    if not image:
        return [[]]

    row, col = len(image), len(image[0])
    prevcolor = image[sr][sc]
    if prevcolor == newColor:
        return image
    queue = [(sr, sc)]

    while queue:
        i, j = queue.pop(0)

        image[i][j] = newColor

        for cor in [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]:
            if 0 <= cor[0] < row and 0 <= cor[1] < col and image[cor[0]][cor[1]] == prevcolor:
                queue.append(cor)

    return image


# ! 其实做下面3道题首要需要考虑的点是如何根据已知的序列得到左子树和右子树

# %%
# * Construct Binary Tree from Preorder and Inorder Traversal
def buildTree(self, preorder, inorder):
    """
    :type preorder: List[int]
    :type inorder: List[int]
    :rtype: TreeNode
    """
    # ! 利用了递归的方式进行求解
    if not preorder or not inorder:
        return None

    rootValue = preorder[0]

    root = TreeNode(rootValue)
    root_indx = inorder.index(rootValue)

    root.left = self.buildTree(preorder[1:root_indx + 1], inorder[:root_indx])
    root.right = self.buildTree(
        preorder[root_indx + 1:], inorder[root_indx + 1:])

    return root


# %%
# * Construct Binary Tree from Inorder and Postorder Traversal
def buildTree(self, inorder, postorder):
    """
    :type inorder: List[int]
    :type postorder: List[int]
    :rtype: TreeNode
    """
    if not postorder or not inorder:
        return None

    rootValue = postorder[-1]

    root = TreeNode(rootValue)
    root_indx = inorder.index(rootValue)

    root.right = self.buildTree(
        inorder[root_indx + 1:], postorder[root_indx:-1])
    root.left = self.buildTree(inorder[:root_indx], postorder[:root_indx])

    return root


# %%
# * Construct Binary Tree from Preorder and Postorder Traversal
def constructFromPrePost(self, pre, post):
    """
    :type pre: List[int]
    :type post: List[int]
    :rtype: TreeNode
    """
    if not pre or not post:
        return None

    rootValue = pre[0]
    root = TreeNode(rootValue)
    if len(pre) == 1:
        return root
    post_indx = post.index(pre[1])

    root.left = self.constructFromPrePost(
        pre[1:post_indx + 2], post[0:post_indx + 1])
    root.right = self.constructFromPrePost(
        pre[post_indx + 2:], post[post_indx + 1:-1])

    return root


# %%
# * Serialize and Deserialize Binary Tree

class Codec:  # ! 全程用bfs解

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ""
        serialized_tree = ""
        bfs = [root]

        while bfs:
            node = bfs.pop(0)

            serialized_tree += "{},".format(node.val if node else "null")

            if node:
                bfs.append(node.left)
                bfs.append(node.right)

        return serialized_tree

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None

        node_list = data.split(",")
        root = TreeNode(node_list.pop(0))
        stack = [root]

        while stack:
            cur_node = stack.pop(0)

            left = node_list.pop(0)
            if left != "null":
                cur_node.left = TreeNode(left)
                stack.append(cur_node.left)
            right = node_list.pop(0)
            if right != "null":
                cur_node.right = TreeNode(right)
                stack.append(cur_node.right)

        return root


# %%
# * Subtract the Product and Sum of Digits of an Integer
def subtractProductAndSum(self, n):
    """
    :type n: int
    :rtype: int
    """
    arr_n = str(n).strip()

    product = 1
    summation = 0

    for s in arr_n:
        product *= int(s)
        summation += int(s)

    return product - summation


# %%
# * Split a String in Balanced Strings
def balancedStringSplit(self, s):
    """
    :type s: str
    :rtype: int
    """
    # ! 使用贪心算法
    if len(s) == 1:
        return 0
    num_l = 0
    num_r = 0
    count = 0

    for i in s:

        if i == "R":
            num_r += 1

        else:
            num_l += 1

        if num_l == num_r:
            count += 1

    return count


# %%
# * Cells with Odd Values in a Matrix
def oddCells(self, n, m, indices):
    """
    :type n: int
    :type m: int
    :type indices: List[List[int]]
    :rtype: int
    """
    initial_array = np.zeros((n, m))

    for r, c in indices:
        initial_array[r, :] += 1
        initial_array[:, c] += 1

    modulo_array = np.array([x % 2 for x in initial_array])
    return int(np.sum(modulo_array))


# %%
# * Remove Outermost Parentheses
def removeOuterParentheses(self, S):
    """
    :type S: str
    :rtype: str
    """
    left_num = 0
    right_num = 0
    left_indx = 0
    res = []
    # ! 其实用不到栈
    for i in range(len(S)):
        if S[i] == "(":
            left_num += 1
        else:
            right_num += 1

        if left_num == right_num:
            res.append(S[left_indx + 1: i])
            left_indx = i + 1
            left_num = 0
            right_num = 0

    return "".join(res)


# %%
# * Unique Number of Occurrences
def uniqueOccurrences(self, arr):
    """
    :type arr: List[int]
    :rtype: bool
    """
    arr_dict = Counter(arr).values()

    return True if len(set(arr_dict)) == len(arr_dict) else False


# %%
# * Merge Two Binary Trees
def mergeTrees(self, t1, t2):
    """
    :type t1: TreeNode
    :type t2: TreeNode
    :rtype: TreeNode
    """
    # ! 使用递归的方法解
    if(t1 == None):
        return t2

    if(t2 == None):
        return t1

    t1.val = t1.val + t2.val

    t1.left = self.mergeTrees(t1.left, t2.left)
    t1.right = self.mergeTrees(t1.right, t2.right)

    return t1


# %%
# * Height Checker
def heightChecker(self, heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    rightOrder_heights = sorted(heights)
    count = 0

    for i in range(len(heights)):
        if rightOrder_heights[i] != heights[i]:
            count += 1

    return count

# %%
# * Find Words That Can Be Formed by Characters


def countCharacters(self, words, chars):
    """
    :type words: List[str]
    :type chars: str
    :rtype: int
    """
    chars_dict = Counter(chars)

    res = 0

    for i in words:
        flag = 0
        tmp = Counter(i)

        for k in tmp:
            if not k in chars or tmp[k] > chars_dict[k]:
                flag = 1
                break

        if flag == 0:
            res += len(i)

    return res


# %%
# * Relative Sort Array
def relativeSortArray(self, arr1, arr2):
    """
    :type arr1: List[int]
    :type arr2: List[int]
    :rtype: List[int]
    """
    # ! 利用 python 字典的get方法以及sorted函数进行求解
    k = {b: i for i, b in enumerate(arr2)}

    return sorted(arr1, key=lambda a: k.get(a, 1000 + a))


# %%
# * Minimum Absolute Difference
def minimumAbsDifference(self, arr):
    """
    :type arr: List[int]
    :rtype: List[List[int]]
    """
    # ! 先排序再遍历
    new_arr = sorted(arr)
    min_val = float("inf")
    res = []

    for i in range(len(new_arr) - 1):
        tmp = new_arr[i + 1] - new_arr[i]

        if tmp == min_val:
            res.append([new_arr[i], new_arr[i + 1]])

        if tmp < min_val:
            min_val = tmp
            while res:
                res.pop(0)
            res.append([new_arr[i], new_arr[i + 1]])

    return res


# %%
# * Remove All Adjacent Duplicates In String

def removeDuplicates(self, S):
    """
    :type S: str
    :rtype: str
    """
    stack = []

    for i in S:
        if stack and stack[-1] == i:
            stack.pop()
            continue

        stack.append(i)

    return "".join(stack)


# %%
# * Sum of Root To Leaf Binary Numbers
def sumRootToLeaf(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    # ! 利用深度优先搜索，寻找到每条路径
    self.res = []

    tmp_str = ""

    def dfs(node, tmp):
        if node:

            tmp += str(node.val)

            if not node.left and not node.right:
                self.res.append(tmp)
                tmp = ""

            if node.left:
                dfs(node.left, tmp)

            if node.right:
                dfs(node.right, tmp)

    dfs(root, tmp_str)

    final_res = 0

    for i in self.res:
        final_res += int(i, 2)

    return final_res


# %%
# * Occurrences After Bigram
def findOcurrences(self, text, first, second):
    """
    :type text: str
    :type first: str
    :type second: str
    :rtype: List[str]
    """
    text_arr = text.split(" ")
    res = []

    if len(text_arr) < 3:
        return []

    for i in range(len(text_arr) - 2):
        if text_arr[i] == first and text_arr[i + 1] == second:
            res.append(text_arr[i + 2])

    return res


# %%
# * Leaf-Similar Trees
def leafSimilar(self, root1, root2):
    """
    :type root1: TreeNode
    :type root2: TreeNode
    :rtype: bool
    """

    self.res1 = []
    self.res2 = []

    def dfs(node, res):
        if node:

            if not node.left and not node.right:
                res.append(node.val)

            if node.left:
                dfs(node.left, res)

            if node.right:
                dfs(node.right, res)

    dfs(root1, self.res1)
    dfs(root2, self.res2)

    return True if self.res1[:] == self.res2[:] else False


# %%
# * Last Stone Weight
def lastStoneWeight(self, stones):
    """
    :type stones: List[int]
    :rtype: int
    """

    while len(stones) > 1:
        stones.sort()

        if stones[-1] == stones[-2]:
            del stones[-1]
            del stones[-1]

        else:
            stones = [stones[-1] - stones[-2]] + stones
            del stones[-1]
            del stones[-1]

    return 0 if not stones else stones[0]


# %%
# * Complement of Base 10 Integer
def bitwiseComplement(self, N):
    """
    :type N: int
    :rtype: int
    """
    str_n = bin(N)[2:]

    new_n = ""

    for item in str_n:
        if item == "0":
            new_n += "1"

        else:
            new_n += "0"

    return int(new_n, 2)


# %%
# * Element Appearing More Than 25% In Sorted Array
def findSpecialInteger(self, arr):
    """
    :type arr: List[int]
    :rtype: int
    """
    return Counter(arr).most_common(1)[0][0]


# %%
# * Maximum Number of Balloons
def maxNumberOfBalloons(self, text):
    """
    :type text: str
    :rtype: int
    """
    text_dict = Counter(text)

    main_dict = Counter("balloon")
    flag = 0
    res = []

    for k in text_dict.keys():
        if k in "balloon":
            flag += 1
            res.append(text_dict[k] // main_dict[k])

    return min(res) if flag == 5 else 0


# %%
# * Duplicate Zeros
def duplicateZeros(self, arr):
    """
    :type arr: List[int]
    :rtype: None Do not return anything, modify arr in-place instead.
    """

    i = 0

    while i < len(arr):

        if arr[i] == 0:
            arr.insert(i + 1, 0)
            arr.pop()
            i += 2
        else:
            i += 1


# %%
# * Compare Strings by Frequency of the Smallest Character
def numSmallerByFrequency(self, queries, words):
    """
    :type queries: List[str]
    :type words: List[str]
    :rtype: List[int]
    """
    # ! N平方的复杂度 480ms，有点耗时
    count_res = []

    query_res = [item.count(sorted(item)[0]) for item in queries]
    word_res = [item.count(sorted(item)[0]) for item in words]

    for i in query_res:
        tmp = 0

        for j in word_res:
            if i < j:
                tmp += 1

        count_res.append(tmp)

    return count_res


# %%
# * N-th Tribonacci Number
def tribonacci(self, n):
    """
    :type n: int
    :rtype: int
    """
    # ! 递归的方法超时了
    def cal_fib(n):
        if n == 0:
            return 0

        if n == 1:
            return 1

        if n == 2:
            return 1

        return cal_fib(n - 1) + cal_fib(n - 2) + cal_fib(n - 3)

    return cal_fib(n)


def tribonacci(self, n):
    """
    :type n: int
    :rtype: int
    """
    # ! 这种方法快的原因是在递归的过程中，会重复计算一些值，所以利用cache字典将相应的值保存下来
    cache = {}

    def cal_fib(n):
        if n in cache:
            return cache[n]  # ! key step

        if n < 3:
            if n == 0:
                res = 0

            if n == 1:
                res = 1

            if n == 2:
                res = 1
        else:
            res = cal_fib(n - 1) + cal_fib(n - 2) + cal_fib(n - 3)

        cache[n] = res  # ! key step

        return res

    return cal_fib(n)


# %%
# * Fair Candy Swap
def fairCandySwap(self, A, B):
    """
    :type A: List[int]
    :type B: List[int]
    :rtype: List[int]
    """
    # ! 列方程解题
    sum1, sum2 = sum(A), sum(B)
    B = set(B)
    for one in A:
        temp = int((sum2+2*one-sum1)/2)
        if temp in B:
            return [one, temp]


# %%
# * Count Binary Substrings
def countBinarySubstrings(self, s):
    """
    :type s: str
    :rtype: int
    """
    prelen = 0
    curlen = 1
    count = 0
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            curlen += 1
        else:
            prelen = curlen
            curlen = 1
        if prelen >= curlen:
            count += 1

    return count


# %%
# * Convert Integer to the Sum of Two No-Zero Integers
def getNoZeroIntegers(self, n):
    """
    :type n: int
    :rtype: List[int]
    """
    if n == 2:
        return [1, 1]
    mid = n // 2
    for i in range(1, mid):
        if '0' in str(i):
            continue
        tmp = n - i

        if '0' not in str(tmp):
            return [i, tmp]


# %%
# * Two City Scheduling
def twoCitySchedCost(self, costs):
    """
    :type costs: List[List[int]]
    :rtype: int
    """
    # ! 计算cost[0] - cost[1]是为了得到每步选择的收益，而且收益要从最大的开始选择（贪心算法）
    res = 0
    person_num = len(costs) // 2
    count_a = 0
    count_b = 0
    # ! 重要
    for person in sorted(costs, key=lambda cost: -abs(cost[0] - cost[1])):
        if (person[0] < person[1] and count_a < person_num) or count_b == person_num:
            res += person[0]
            count_a += 1

        else:
            res += person[1]
            count_b += 1

        return res


# %%
# * Reformat The String
def reformat(self, s):
    """
    :type s: str
    :rtype: str
    """
    str_arr = []
    di_arr = []

    for c in s:
        if ord(c) >= ord('0') and ord(c) <= ord('9'):
            di_arr.append(c)
        else:
            str_arr.append(c)

    s = len(str_arr)
    d = len(di_arr)

    if s == d or abs(s - d) == 1:

        res = ''

        for i in str_arr:
            res += i
            if di_arr:
                res += di_arr[0]
                di_arr.pop(0)

        if di_arr:
            return di_arr[0] + res
        else:
            return res

    else:
        return ""


# %%
# * Running Sum of 1d Array
def runningSum(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    res = []

    for i in range(len(nums)):
        if not res:
            res.append(nums[i])
        else:
            res.append(res[-1] + nums[i])

    return res


# %%
# * Number of Good Pairs
def numIdenticalPairs(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    res = 0
    num_counter = Counter(nums)

    for item in num_counter.items():
        res += (item[1] * (item[1] - 1)) / 2

    return res


# %%
# * Shuffle the Array
def shuffle(self, nums, n):
    """
    :type nums: List[int]
    :type n: int
    :rtype: List[int]
    """
    res = []
    x = nums[:n]
    y = nums[n:]
    i = 0

    while len(res) < len(nums):

        res.append(x[i])
        res.append(y[i])
        i += 1

    return res


# %%
# * Number of Steps to Reduce a Number to Zero
def numberOfSteps(self, num):
    """
    :type num: int
    :rtype: int
    """
    steps = 0
    while num != 0:
        if num % 2 == 0:
            num = num / 2
        else:
            num = num - 1

        steps += 1

    return steps


# %%
# * How Many Numbers Are Smaller Than the Current Number
def smallerNumbersThanCurrent(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    dic = {}
    res = []
    nums_sort = sorted(nums)

    for i in range(len(nums_sort)):
        if i == 0:
            dic[nums_sort[i]] = 0
        else:
            if nums_sort[i] == nums_sort[i - 1]:
                dic[nums_sort[i]] = dic[nums_sort[i - 1]]
            else:
                dic[nums_sort[i]] = i

    for num in nums:
        res.append(dic[num])

    return res


# %%
# * XOR Operation in an Array
def xorOperation(self, n, start):
    """
    :type n: int
    :type start: int
    :rtype: int
    """
    i = 1
    num = start
    while i <= n - 1:
        tmp = start + 2 * i
        num ^= tmp
        i += 1

    return num


# %%
# * Decompress Run-Length Encoded List
def decompressRLElist(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    i = 0
    res = []

    while i < len(nums):
        fre = nums[i]
        val = nums[i + 1]

        res += [val] * fre

        i += 2

    return res


# %%
# * Create Target Array in the Given Order
def createTargetArray(self, nums, index):
    """
     :type nums: List[int]
     :type index: List[int]
     :rtype: List[int]
     """
    target = []
    for i in range(len(nums)):
        if len(target) - 1 < index[i]:
            target.append(nums[i])

        else:
            target.insert(index[i], nums[i])

    return target


# %%
# * Find Numbers with Even Number of Digits
def findNumbers(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    count = 0
    
    for num in nums:
        if len(str(num)) % 2 == 0:
            count += 1
    
    
    return count


# %%
# * Convert Binary Number in a Linked List to Integer
def getDecimalValue(self, head):
    """
    :type head: ListNode
    :rtype: int
    """
    str_h = "" + str(head.val)
    
    while head.next:
        head = head.next
        str_h += str(head.val)
    
    return int(str_h, 2)

# %%
# * Shuffle String
def restoreString(self, s, indices):
    """
    :type s: str
    :type indices: List[int]
    :rtype: str
    """
    # ! 利用插入排序的思想进行解题
    s = list(s)
    for i in range(1, len(s)):
        j = 0
        while j < i:
            if indices[j] >= indices[i]:
                indices.insert(j, indices[i])
                del indices[i + 1]
                s.insert(j, s[i])
                del s[i + 1]
            j += 1
    
    return "".join(s)


# %%
# * Minimum Time Visiting All Points
def minTimeToVisitAllPoints(self, points):
    """
    :type points: List[List[int]]
    :rtype: int
    """
    # ! 先尽量走斜线，再走直线
    time = 0
    for i in range(len(points) - 1):
        x = abs(points[i][0] - points[i + 1][0])
        y = abs(points[i][1] - points[i + 1][1])
        
        if x == 0:
            time += y
            continue
        
        if y == 0:
            time += x
            continue
        
        if x == y:
            time += y
        elif x < y:
            time += y
        
        else:
            time += x
    
    
    return time


# %%
# * Maximum 69 Number
def maximum69Number (self, num):
    """
    :type num: int
    :rtype: int
    """
    s_num = list(str(num))
    max_num = num
    
    for i in range(len(s_num)):
        if s_num[i] == "6":
            s_num[i] = "9"
            if int("".join(s_num)) > max_num:
                max_num = int("".join(s_num))
                return max_num
            
    
    
    return max_num


# %%
# * Maximum Product of Two Elements in an Array
def maxProduct(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    nums.sort()
    
    return (nums[-1] - 1) * (nums[-2] - 1)


# %%
# * Number of Students Doing Homework at a Given Time
def busyStudent(self, startTime, endTime, queryTime):
    """
    :type startTime: List[int]
    :type endTime: List[int]
    :type queryTime: int
    :rtype: int
    """
    count = 0
    for i in range(len(startTime)):
        if queryTime >= startTime[i] and queryTime <= endTime[i]:
            count += 1
    
    
    return count


# %%
# * Decrypt String from Alphabet to Integer Mapping

def freqAlphabets(self, s):
    """
    :type s: str
    :rtype: str
    """
    table = {'1': 'a', '2': 'b','3': 'c','4': 'd','5': 'e','6': 'f','7': 'g','8': 'h','9': 'i','10#': 'j','11#': 'k','12#': 'l','13#': 'm','14#': 'n','15#': 'o','16#': 'p','17#': 'q','18#': 'r','19#': 's','20#': 't','21#': 'u','22#': 'v','23#': 'w','24#': 'x','25#': 'y','26#': 'z',}
    
    s_arr = s.split("#")
    code = []
    res = ""
    
    for item in s_arr[0:-1]:
        if item and len(item) < 2:
            code.append(item)
            continue

        if item and len(item) == 2:
            code.append(item + '#')
            continue
        
        if item and len(item) > 2:
            for i in range(len(item) - 2):
                code.append(item[i])
            
            code.append(item[-2:] + '#')
    
    
    for i in s_arr[-1]:
        code.append(i)
    
    
    for c in code:
        res += table[c]
    
    
    return res

# %%
# * Count Negative Numbers in a Sorted Matrix
def countNegatives(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    # ! 暴力求解，O(n**2)
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] < 0:
                count += 1
    
    return count


# %%
# * Replace Elements with Greatest Element on Right Side
def replaceElements(self, arr):
    """
    :type arr: List[int]
    :rtype: List[int]
    """
    # ! 暴力求解
    for i in range(len(arr)):
        if i == len(arr) - 1:
            arr[i] = -1
            break
        
        arr[i] = max(arr[i + 1:])
    
    
    return arr


def replaceElements(self, arr):
    """
    :type arr: List[int]
    :rtype: List[int]
    """
    # ! 对数组进行倒序遍历，每次都更新maximum的值
    # ! 倒序
    res = []
    maximum = arr[-1]
    for i in range(len(arr) - 1, -1, -1): # !
        if i == len(arr) - 1:
            res.insert(0, -1)
            continue
        
        res.insert(0, maximum)
        
        if arr[i] >= maximum:
            maximum = arr[i]
        
    return res


# %%
# * Generate a String With Characters That Have Odd Counts
def generateTheString(self, n):
    """
    :type n: int
    :rtype: str
    """
    # ! 这题无语
    if n % 2 == 0:
        
        return 'a' * (n - 1) + 'b'
    
    else:
        return n * 'a'


# %%
# * Final Prices With a Special Discount in a Shop
def finalPrices(self, prices):
    """
    :type prices: List[int]
    :rtype: List[int]
    """
    # ! 暴力求解
    res = []
    flag = 0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            if prices[j] <= prices[i]:
                flag = 1
                res.append(prices[i] - prices[j])
                break
        
        if flag == 0:
            res.append(prices[i])
        else:
            flag = 0
    
    
    return res


# %%
# * Make Two Arrays Equal by Reversing Sub-arrays
def canBeEqual(self, target, arr):
    """
    :type target: List[int]
    :type arr: List[int]
    :rtype: bool
    """
    # ! 无语
    return sorted(target) == sorted(arr)


def canBeEqual(self, target, arr):
    """
    :type target: List[int]
    :type arr: List[int]
    :rtype: bool
    """
    # ! 这种方法也可以
    for num in arr:
        if num not in target:
            return False
        else:
            target.remove(num)
        
    
    
    return True


# %%
# * Can Make Arithmetic Progression From Sequence
def canMakeArithmeticProgression(self, arr):
    """
    :type arr: List[int]
    :rtype: bool
    """
    arr_n = sorted(arr)
    diff = arr_n[1] - arr_n[0]
    
    for i in range(len(arr_n) - 1):
        if (arr_n[i + 1] - arr_n[i]) != diff:
            return False
    
    
    return True


# %%
# * Number of Recent Calls
class RecentCounter(object):

    def __init__(self):
        self.d = []

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        # ! 题意有点莫名其妙
        self.d.append(t)

        while self.d[-1] - self.d[0] > 3000:
            self.d.pop(0)

        return len(self.d)


# %%
# * Increasing Order Search Tree
def increasingBST(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if not root: return []
    self.result = []
    def inorderList(node):

        if node:

            inorderList(node.left)

            self.result.append(node.val)

            inorderList(node.right)
    
    inorderList(root)
    
    n_root = TreeNode(self.result[0])
    self.result.pop(0)
    
    loop = n_root
    
    while self.result:
        node = TreeNode(self.result[0])
        loop.right = node
        loop.left = None
        loop = node
        self.result.pop(0)
    
    return n_root


# %%
# * Build an Array With Stack Operations
def buildArray(self, target, n):
    """
    :type target: List[int]
    :type n: int
    :rtype: List[str]
    """
    op1 = "Push"
    op2 = "Pop"
    res = []
    i = 0
    j = 1
    
    while i < len(target) and j <= n:
        
        if ori[j] == target[i]:
            res.append(op1)
            i += 1
            j += 1
            continue
        
        
        if ori[j] != target[i]:
            res.append(op1)
            res.append(op2)
            j += 1
    
    return res


# %%
# * Sort Integers by The Number of 1 Bits
def sortByBits(self, arr):
    """
    :type arr: List[int]
    :rtype: List[int]
    """
    # ! 借鉴归并排序的思路
    if len(arr) < 2:
        return arr

    n = len(arr) // 2

    left_arr = self.sortByBits(arr[0: n])
    right_arr = self.sortByBits(arr[n:])

    res = []

    while left_arr or right_arr:
        if not left_arr:
            res += right_arr
            break

        if not right_arr:
            res += left_arr
            break

        left = left_arr[0]
        right = right_arr[0]

        left_1 = bin(left).count('1') 
        right_1 = bin(right).count('1') 

        if left_1 < right_1:
            res.append(left)
            left_arr.pop(0)

        elif left_1 > right_1:
            res.append(right)
            right_arr.pop(0)

        else:
            if left <= right:
                res.append(left)
                left_arr.pop(0)

            else:
                res.append(right)
                right_arr.pop(0)

    return 


# %%
# * Find the Distance Value Between Two Arrays
def findTheDistanceValue(self, arr1, arr2, d):
    """
    :type arr1: List[int]
    :type arr2: List[int]
    :type d: int
    :rtype: int
    """
    count = 0
    arr2.sort()
    
    def is_distance(num1,num2, arr):

        left, right = 0, len(arr) - 1

        while left <= right:

            mid = (left + right) // 2

            if arr[mid] <= num1 and arr[mid] >= num2:
                return False
            elif arr[mid] > num1:
                right = mid - 1
            else:
                left = mid + 1
        
        
        return True
    
    
    for num in arr1:
        num1 = num + d
        num2 = num - d
        
        if is_distance(num1, num2, arr2):
            count += 1
    
    
    return count


# %%
# * Find Lucky Integer in an Array
def findLucky(self, arr):
    """
    :type arr: List[int]
    :rtype: int
    """
    maximum = -1
    arr_dict = Counter(arr)
    
    for item in arr_dict.items():
        if item[0] == item[1] and item[0] >= maximum:
            maximum = item[0]
    
    
    return maximum


# %%
# * Check If a Word Occurs As a Prefix of Any Word in a Sentence
def isPrefixOfWord(self, sentence, searchWord):
    """
    :type sentence: str
    :type searchWord: str
    :rtype: int
    """
    # ! 判断前缀
    arr = sentence.split(" ")
    
    
    for i in range(len(arr)):
        if arr[i].startswith(searchWord):
            return i + 1
    
    
    return -1


# %%
# * Find Positive Integer Solution for a Given Equation
def findSolution(self, customfunction, z):
    """
    :type num: int
    :type z: int
    :rtype: List[List[int]]
    """
    res = []
    
    for x in range(1, z + 1):
        for y in range(1, z + 1):
            if customfunction.f(x,y) == z:
                res.append([x, y])
    
    
    return res


# %%
# * Find N Unique Integers Sum up to Zero
def sumZero(self, n):
    """
    :type n: int
    :rtype: List[int]
    """
    # ! 分奇偶进行讨论
    if n == 1: return [0]
    res = []
    if n % 2 == 0:
        for i in range(1, n / 2 + 1):
            res.append(i)
            res.append(-i)
    else:
        for i in range(1, n // 2 + 1):
            res.append(i)
            res.append(-i)
        
        res.append(0)
    
    
    return res


# %%
# * Average Salary Excluding the Minimum and Maximum Salary
def average(self, salary):
    """
    :type salary: List[int]
    :rtype: float
    """
    # ! 这是个智障题目
    salary.sort()
    
    length = len(salary) - 2
    res = 0.0
    
    for i in range(1, len(salary) - 1):
        res += salary[i]
    
    
    res = float(res) / float(length)
    
    return float(res)


# %%
# * Matrix Cells in Distance Order
def allCellsDistOrder(self, R, C, r0, c0):
    """
    :type R: int
    :type C: int
    :type r0: int
    :type c0: int
    :rtype: List[List[int]]
    """
    ori = [r0, c0]
    
    points = []
    res = []
    
    for i in range(R):
        for j in range(C):
            distances = abs(ori[0] - i) + abs(ori[1] - j)
            points.append([i, j, distances])
    
    points = sorted(points, key=lambda x: x[2])
    
    for p in points:
        res.append([p[0], p[1]])
    
    
    return res


# %%
# * Count Largest Group
def countLargestGroup(self, n):
    """
    :type n: int
    :rtype: int
    """
    tmp = defaultdict(list)
    count = 0

    for num in range(1, n + 1):
        key = sum([int(i) for i in str(num)])
        if not tmp or not tmp[key]:
            tmp[key] = [num]
            continue
        
        if tmp and tmp[key]:
            tmp[key].append(num)
    
    tmp = sorted(tmp.items(), key=lambda item: len(item[1]), reverse=True)
    
    maximum = len(tmp[0][1])
    
    for item in tmp:
        if len(item[1]) == maximum:
            count += 1
            continue
        
        if len(item[1]) < maximum:
            break
    
    
    return count


# %%
# * Print In Order
class Foo(object):
    def __init__(self):
        self.f = True
        self.s = False
        self.t = False

    def first(self, printFirst):
        """
        :type printFirst: method
        :rtype: void
        """

        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.s = True

    def second(self, printSecond):
        """
        :type printSecond: method
        :rtype: void
        """

        # printSecond() outputs "second". Do not change or remove this line.
        while 1:
            if self.s:
                break

        printSecond()
        self.t = True

    def third(self, printThird):
        """
        :type printThird: method
        :rtype: void
        """

        # printThird() outputs "third". Do not change or remove this line.
        while 1:
            if self.t:
                break

        printThird()


# %%
# * Minimum Value to Get Positive Step by Step Sum
def minStartValue(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # ! 灵活运用 prefix sum (前缀和)
    acc = 0
    minimum = float('inf')
    
    for num in nums:
        acc += num
        if acc < minimum:
            minimum = acc
    
    
    return 1 if minimum > 0 else abs(minimum) + 1  # ! 关键


# %%
# * String Matching in an Array
def stringMatching(self, words):
    """
    :type words: List[str]
    :rtype: List[str]
    """
    # ! 暴力求解
    res = []
    for i in range(len(words)):
        for j in range(len(words)):
            if i == j:
                continue
            else:
                if words[i] in words[j]:
                    res.append(words[i])
                    break
    
    
    return res


# %%
# * Consecutive Characters
def maxPower(self, s):
    """
    :type s: str
    :rtype: int
    """
    # ! 利用list分段进行存储substring
    res = defaultdict(list)
    
    for i in range(len(s)):
        if i == 0:
            res[s[i]].append(1)
            continue
        if res[s[i]] and s[i - 1] != s[i]:
            res[s[i]].append(1)
            continue
        
        if res[s[i]] and s[i - 1] == s[i]:
            res[s[i]][-1] += 1
            continue
        
        if not res[s[i]]:
            res[s[i]].append(1)
    
    
    return max(sorted(res.items(), key=lambda x: max(x[1]))[-1][1])


# %%
# * Water Bottles
def numWaterBottles(self, numBottles, numExchange):
    """
    :type numBottles: int
    :type numExchange: int
    :rtype: int
    """
    res = numBottles
    
    n = numBottles
    
    while n >= numExchange:
        
        res += n // numExchange
        
        n = n // numExchange + n % numExchange
        
    
    return res


# %%
# * Maximum Score After Splitting a String
def maxScore(self, s):
    """
    :type s: str
    :rtype: int
    """
    # ! 暴力求解
    maximum = 0
    arr_s = list(s)
    for i in range(1, len(arr_s)):
        tmp = arr_s[0:i].count('0') + arr_s[i:].count('1')
        if maximum < tmp:
            maximum = tmp
    
    return maximum


# %%
# * Rank Transform of an Array
def arrayRankTransform(self, arr):
    """
    :type arr: List[int]
    :rtype: List[int]
    """
    # ! 利用hash table进行解题
    arr_sort = sorted(arr)
    da = defaultdict(int)
    
    for i in range(len(arr_sort)):
        if i == 0:
            da[arr_sort[i]] = 1
            continue
        
        if not da[arr_sort[i]]:
            da[arr_sort[i]] = da[arr_sort[i - 1]] + 1
    
    return [da[num] for num in arr]


# %%
# *  Check If N and Its Double Exist
def checkIfExist(self, arr):
    """
    :type arr: List[int]
    :rtype: bool
    """
    for i in range(len(arr)):
        if (arr[i] % 2 == 0 and arr[i] / 2 in arr[i + 1:]) or arr[i] * 2 in arr[i + 1:]:
            return True
    
    
    return False


# %%
# * Check If It Is a Straight Line
def checkStraightLine(self, coordinates):
    """
    :type coordinates: List[List[int]]
    :rtype: bool
    """
    # ! 分情况讨论，利用直线方程进行求解
    if coordinates[0][0] == coordinates[1][0]:
        for point in coordinates[2:]:
            if point[0] != coordinates[0][0]:
                return False

    else: 
        line_a = float(coordinates[0][1] - coordinates[1][1]) / float(coordinates[0][0] - coordinates[1][0])
        line_b = float(coordinates[0][1] - line_a * coordinates[0][0])
    
        for point in coordinates[2:]:
            if float(point[0] * line_a + line_b) != float(point[1]):
                return False
    
    return True


# %%
# * Deepest Leaves Sum
def deepestLeavesSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    # !利用广度优先遍历进行求解
    if not root: return None
    queue = [root]
    res = 0
    count = 0
    lastnum = 1
    tmp = 0
    
    while queue:
        node = queue.pop(0)
        tmp += node.val
        lastnum -= 1
        
        if node.left:
            queue.append(node.left)
            count += 1
        
        if node.right:
            queue.append(node.right)
            count += 1
        
        if lastnum == 0:
            res = tmp
            tmp = 0
            lastnum = count
            count = 0
    
    
    return res


# %%
# * Count Number of Teams
def numTeams(self, rating):
    """
    :type rating: List[int]
    :rtype: int
    """
    # ! 暴力求解
    count = 0
    for i in range(len(rating)):
        for j in range(i, len(rating)):
            for k in range(j, len(rating)):
                if (rating[i] < rating[j] and rating[j] < rating[k]) or (rating[i] > rating[j] and rating[j] > rating[k]):
                    count += 1
    
    
    return count


# %%
# * Queries on a Permutation With Key
def processQueries(self, queries, m):
    """
    :type queries: List[int]
    :type m: int
    :rtype: List[int]
    """
    # ! 这题有点无聊
    p = [i for i in range(1,m + 1)]
    res = []
    for num in queries:
        tmp = p.index(num)
        res.append(tmp)
        del p[tmp]
        p.insert(0, num)
    
    
    return res


# %%
# * Group the People Given the Group Size They Belong To
def groupThePeople(self, groupSizes):
    """
    :type groupSizes: List[int]
    :rtype: List[List[int]]
    """
    di = defaultdict(list)
    res = []
    
    for i in range(len(groupSizes)):
        if not di[groupSizes[i]] or (di[groupSizes[i]] and len(di[groupSizes[i]]) < groupSizes[i]):
            di[groupSizes[i]].append(i)
            if len(di[groupSizes[i]]) == groupSizes[i]:
                res.append(di[groupSizes[i]])
                di[groupSizes[i]] = []
            continue
        
        if di[groupSizes[i]] and len(di[groupSizes[i]]) == groupSizes[i]:
            res.append(di[groupSizes[i]])
            di[groupSizes[i]] = []
            di[groupSizes[i]].append(i)
    
    
    return res


# %%
# * Find a Corresponding Node of a Binary Tree in a Clone of That Tree
def getTargetCopy(self, original, cloned, target):
    """
    :type original: TreeNode
    :type cloned: TreeNode
    :type target: TreeNode
    :rtype: TreeNode
    """
    # ! 分情况判断即可
    queue = [cloned]
    
    while queue:
        node = queue.pop(0)
        if node.val == target.val:
            if not node.left and not target.left and not node.right and not target.right:
                return node
            if node.left and target.left and node.left.val == target.left.val and node.right and target.right and node.right.val == target.right.val:
                return node
            
            if node.left and target.left and not node.right and not target.right and node.left.val == target.left.val:
                return node
            
            if node.right and target.right and node.right.val == target.right.val and not node.left and not target.left:
                return node
            
        if node.left:
            queue.append(node.left)
        
        if node.right:
            queue.append(node.right)
    
    
    return None


# %%
# * Binary Search Tree to Greater Sum Tree
def bstToGst(self, root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    if not root: return []
    acc = 0
    def right_order_tra(acc,node):
        if not node:
            return 0
        
        if not node.right and not node.left:
            acc += node.val
            node.val = acc
            return acc
        
        if not node.right and node.left:
            acc += node.val
            node.val = acc
            left_acc = right_order_tra(acc, node.left)
            return left_acc

        
        right_acc = right_order_tra(acc, node.right)
        node.val += right_acc
        right_acc = node.val
        if not node.left:
            return right_acc
        
        left_acc = right_order_tra(right_acc, node.left)
        
        return left_acc
    
    
    right_order_tra(acc, root)
    
    return root


# %%
# * Design a Stack With Increment Operation
class CustomStack(object):

    def __init__(self, maxSize):
        """
        :type maxSize: int
        """
        self.length = maxSize
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if len(self.stack) < self.length:
            self.stack.append(x)

    def pop(self):
        """
        :rtype: int
        """
        if self.stack:
            tmp = self.stack.pop()
            return tmp
        else:
            return -1

    def increment(self, k, val):
        """
        :type k: int
        :type val: int
        :rtype: None
        """
        if k < len(self.stack):
            for i in range(k):
                self.stack[i] += val
        else:
            for i in range(len(self.stack)):
                self.stack[i] += val


# %%
# * All Elements in Two Binary Search Trees
def getAllElements(self, root1, root2):
    """
    :type root1: TreeNode
    :type root2: TreeNode
    :rtype: List[int]
    """
    
    def inorder_tra(tmp, node):
        if node:
            inorder_tra(tmp, node.left)
            tmp.append(node.val)
            inorder_tra(tmp, node.right)
        
        return tmp
    
    r1 = []
    r2 = []
    r1 = inorder_tra(r1, root1)
    r2 = inorder_tra(r2, root2)
    
    res = []
    # ! 用归并排序的合并方式进行合并
    while r1 or r2:
        if not r1:
            res += r2
            break

        if not r2:
            res += r1
            break

        left = r1[0]
        right = r2[0]

        if left <= right:
            res.append(left)
            r1.pop(0)

        else:
            res.append(right)
            r2.pop(0)
    return res


# %%
# * Letter Tile Possibilities
def numTilePossibilities(self, tiles):
    """
    :type tiles: str
    :rtype: int
    """
    res = 0
    for i in range(1, len(tiles) + 1):
        
        res += len(list(set(permutations(list(tiles), i))))
    
    
    return res


# %%
# * Minimum Number of Steps to Make Two Strings Anagram
def minSteps(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: int
    """
    s_dict = Counter(s)
    t_dict = Counter(t)
    step = 0
    
    if s_dict == t_dict:
        return step
    
    for key in s_dict:
        if s_dict[key] > t_dict[key]:
            step += s_dict[key] - t_dict[key] 

    
    
    return step

# %%
# * Delete Leaves With a Given Value
def removeLeafNodes(self, root, target):
    """
    :type root: TreeNode
    :type target: int
    :rtype: TreeNode
    """
    # ! 好题
    # ! 从根节点开始一路递归至叶节点，进行判断，然后再回退进行判断
    def dfs(node, parent, flag, target):
        if not node:
            return
        
        dfs(node.left, node,"L", target)
        dfs(node.right, node, "R", target)
        
        
        if not node.left and not node.right and target == node.val:
            if flag =="L":
                parent.left = None  # ! 删除操作
            else:
                parent.right = None
    
    
    if not root:
        return root
    
    dfs(root.left, root, "L", target)
    dfs(root.right, root,"R", target)
    
    if not root.left and not root.right and root.val == target:
        return None
    else:
        return root


# %%
# * Maximum Level Sum of a Binary Tree
def maxLevelSum(self, root):
    """
    :type root: TreeNode
    :rtype: int
    """
    # ! 利用广度优先遍历解题
    if not root:
        return 0
    bfs = [root]
    cur_le = 1
    sum_val = [-float('inf'), 0]
    tmp = 0
    lastnum = 1
    count = 0
    

    while bfs:
        node = bfs.pop(0)
        tmp += node.val
        lastnum -= 1

        if node.left:
            bfs.append(node.left)
            count += 1

        if node.right:
            bfs.append(node.right)
            count += 1

        if lastnum == 0:
            lastnum = count
            count = 0
            if tmp > sum_val[0]:
                sum_val = [tmp, cur_le]
                
            cur_le += 1
            tmp = 0

    return sum_val[1]


# %%
# * Minimum Operations to Make Array Equal
def minOperations(self, n):
    """
    :type n: int
    :rtype: int
    """
    # ! 奇等差数列
    step = 0
    
    
    for i in range(n // 2):
        tmp = 2 * i + 1
        
        step += n - tmp
    
    
    return step

# %%
# * Design Browser History
class BrowserHistory:

    def __init__(self, homepage: str):
        self.q=[homepage]
        self.i=0
    def visit(self, url: str) -> None:
        self.q=self.q[:self.i+1]
        self.i+=1
        self.q.append(url)
    def back(self, steps: int) -> str:
        q=self.q
        self.i= max(self.i-steps,0)
        return q[self.i]
    def forward(self, steps: int) -> str:
        q=self.q
        self.i=min(self.i+steps,len(q)-1)
        return q[self.i]


# %%
# * XOR Queries of a Subarray
def xorQueries(self, arr, queries):
    """
    :type arr: List[int]
    :type queries: List[List[int]]
    :rtype: List[int]
    """
    # ! 暴力求解，超时了
    res = []
    def cal_xor(low, high, arr):
        tmp = 0
        for i in range(low, high + 1):
            tmp ^= arr[i]
        
        return tmp
    
    
    for item in queries:
        res.append(cal_xor(item[0],item[1], arr))
    
    return res



def xorQueries(self, arr, queries):
    """
    :type arr: List[int]
    :type queries: List[List[int]]
    :rtype: List[int]
    """
    # ! 利用前缀和进行求解
    perfix_sum = []
    tmp = 0
    res = []
    for i in range(len(arr)):
        tmp ^= arr[i]
        perfix_sum.append(tmp)
        
    
    for item in queries:
        if item[0] == 0:
            res.append(perfix_sum[item[1]])
            
        elif item[0] == item[1]:
            res.append(arr[item[1]])
        else:
            res.append(perfix_sum[item[1]] ^ perfix_sum[item[0] - 1])
    
    return res

# %%
# * Number of Good Ways to Split a String
def numSplits(self, s):
    """
    :type s: str
    :rtype: int
    """
    # ! 暴力求解，超时
    s_arr = list(s)
    count = 0
    for i in range(1, len(s)):
        if len(Counter(s_arr[0:i]).keys()) == len(Counter(s_arr[i:]).keys()):
            count += 1
    
    
    return count


def numSplits(self, s):
    """
    :type s: str
    :rtype: int
    """
    # ! 利用双字典进行求解
    left_dict = defaultdict(int)
    right_dict = defaultdict(int)
    count = 0
    
    for i in s:
        if not right_dict[i]:
            right_dict[i] = 1
        else:
            right_dict[i] += 1
    
    
    for i in s:
        if not left_dict[i]:
            left_dict[i] = 1
        else:
            left_dict[i] += 1
        
        
        right_dict[i] -= 1

        
        if right_dict[i] == 0:
            del right_dict[i]
        
        if len(right_dict.items()) == len(left_dict.items()):
            count += 1  
    
    
    return count


# %%
# * Range Sum of Sorted Subarray Sums
def rangeSum(self, nums, n, left, right):
    """
    :type nums: List[int]
    :type n: int
    :type left: int
    :type right: int
    :rtype: int
    """
    # ! 暴力求解
    modulo_num = 10**9 + 7
    all_sums = []
    
    for i in range(n):
        tmp = 0
        for j in range(i, n):
            tmp += nums[j]
            all_sums.append(tmp)
    
    
    return int(sum(sorted(all_sums)[left - 1:right]) % modulo_num)


# %%
# * The kth Factor of n
def kthFactor(self, n, k):
    """
    :type n: int
    :type k: int
    :rtype: int
    """
    # ! 简单题，直接按照题意求解即可
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
        
        if count == k:
            return i
    
    return -1


# %%
# * Reduce Array Size to The Half
def minSetSize(self, arr):
    """
    :type arr: List[int]
    :rtype: int
    """
    # ! 利用贪心算法求解
    size = len(arr) / 2
    res = 0
    
    arr_items = sorted(Counter(arr).items(), key=lambda x: x[1])[::-1]
    
    tmp = 0
    
    for item in arr_items:      
        if tmp >= size:
            break
        
        if tmp < size:
            res += 1
            tmp += item[1]
    
    
    return res


# %%
# * Delete Nodes And Return Forest
def delNodes(self, root, to_delete):
    """
    :type root: TreeNode
    :type to_delete: List[int]
    :rtype: List[TreeNode]
    """
    # ! 好题
    # !利用深度优先遍历进行求解
    if not root:
        return []

    self.res = []

    def dfs(node, parent, flag, target):
        if not node:
            return

        dfs(node.left, node,"L", target)
        dfs(node.right, node, "R", target)
        
        if node.left and not node.right and node.val in target:
            if flag == "L":
                parent.left = None
                self.res.append(node.left)
            else:
                parent.right = None
                self.res.append(node.left)
            
            return
        
        if node.right and not node.left and node.val in target:
            if flag == "L":
                parent.left = None
                self.res.append(node.right)
            else:
                parent.right = None
                self.res.append(node.right)
            
            return
        
        if node.left and node.right and node.val in target:
            if flag == "L":
                parent.left = None
                self.res.append(node.left)
                self.res.append(node.right)
            else:
                parent.right = None
                self.res.append(node.left)
                self.res.append(node.right)
            
            return


        if not node.left and not node.right and node.val in target:
            if flag =="L":
                parent.left = None
            else:
                parent.right = None



    dfs(root.left, root, "L", to_delete)
    dfs(root.right, root,"R", to_delete)
    
    
    if root.val in to_delete:
        if root.left and root.right:
            self.res.append(root.left)
            self.res.append(root.right)
        
        if root.left and not root.right:
            self.res.append(root.left)
        
        if root.right and not root.left:
            self.res.append(root.right)
        
    else:
        self.res.append(root)
    
    
    return self.res


# %%
# * Sort Integers by The Power Value
def getKth(self, lo, hi, k):
    """
    :type lo: int
    :type hi: int
    :type k: int
    :rtype: int
    """
    # ! 利用动态规划（DP）和sorted函数进行求解
    self.dp = defaultdict(int)
    
    self.dp[1] = 0
    
    def getPower(num):
        if num in self.dp:
            return self.dp[num]
        
        if num % 2 == 0:
            self.dp[num] = getPower(num // 2) + 1
            
        else:
            self.dp[num] = getPower(num *3 + 1) + 1
        
        
        return self.dp[num]
    
    def power_cmp(x, y):
        xp = getPower(x)
        yp = getPower(y)
        
        if xp == yp:
            return x - y
        
        return xp - yp
    
    
    res = list(range(lo, hi + 1))
    
    return sorted(res, cmp=(power_cmp))[k - 1]


# %%
# * Queens That Can Attack the King
def queensAttacktheKing(self, queens, king):
    """
    :type queens: List[List[int]]
    :type king: List[int]
    :rtype: List[List[int]]
    """
    # ! 对king的八个方向进行穷举检测
    
    res = []
    
    # ! up
    x = king[0]
    y = king[1]
    while x >= 0:
        x -= 1
        if [x, y] in queens:
            res.append([x, y])
            break

    # ! down
    x = king[0]
    y = king[1]
    while x <= 8:
        x += 1
        if [x, y] in queens:
            res.append([x, y])
            break
    # ! right
    x = king[0]
    y = king[1]
    while y <= 8:
        y += 1
        if [x, y] in queens:
            res.append([x, y])
            break
    # ! left
    x = king[0]
    y = king[1]
    while y >= 0:
        y -= 1
        if [x, y] in queens:
            res.append([x, y])
            break
    
    # ! left-up
    x = king[0]
    y = king[1]
    while y >= 0 and x >= 0:
        y -= 1
        x -= 1
        if [x, y] in queens:
            res.append([x, y])
            break
            
    # ! right-down
    x = king[0]
    y = king[1]
    while y <= 8 and x <= 8:
        y += 1
        x += 1
        if [x, y] in queens:
            res.append([x, y])
            break
    
    # ! up-right
    x = king[0]
    y = king[1]
    while y <= 8 and x >= 0:
        y += 1
        x -= 1
        if [x, y] in queens:
            res.append([x, y])
            break
    
    # ! down-left
    x = king[0]
    y = king[1]
    while y >= 0 and x <= 8:
        y -= 1
        x += 1
        if [x, y] in queens:
            res.append([x, y])
            break
    
    return res


# %%
# * Sort an Array
def sortArray(self, nums):
    """
    :type nums: List[int]
    :rtype: List[int]
    """
    return sorted(nums)


# %%
# * Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold
def numOfSubarrays(self, arr, k, threshold):
    """
    :type arr: List[int]
    :type k: int
    :type threshold: int
    :rtype: int
    """
    # ! 进行窗口扫描，并滑动固定大小的窗口
    low = 0
    high = k - 1
    n = len(arr)
    count = 0


    sum_ini = sum(arr[low: high + 1])
    
    if sum_ini / k >= threshold:
        count += 1
    
    if n == k:
        return count
    
    
    if low == high:
        for i in range(1, n):
            if arr[i] >= threshold:
                count += 1
    
    else:
    
        for i in range(1, n - k + 1):
            sum_ini -= arr[i - 1]
            sum_ini += arr[i + k - 1]
            if sum_ini / k >= threshold:
                count += 1  
    
    
    return count


# %%
# * Check If All 1's Are at Least Length K Places Away
def kLengthApart(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    # ! 比较前后位置的"1"
    last_pos = -1
    
    cur_pos = -1
    
    for i in range(len(nums)):
        if nums[i] == 1:
            cur_pos = i
            if last_pos != -1:
                if cur_pos - last_pos - 1 < k:
                    return False
            
            last_pos = cur_pos
    
    return True


# %%
# * Circular Permutation in Binary Representation
def circularPermutation(self, n, start):
    """
    :type n: int
    :type start: int
    :rtype: List[int]
    """
    # ! 生成格雷码序列
    return [start ^ i ^ i >> 1 for i in range(1 << n)]


# %%
# * Minimum Remove to Make Valid Parentheses
def minRemoveToMakeValid(self, s):
    """
    :type s: str
    :rtype: str
    """
    stack = []
    rev = []
    res = list(s)
    
    for i in range(len(s)):
        if s[i] == "(":
            stack.append(i)
            continue
            
        if s[i] == ")" and(stack and stack[-1] >= 0):
            stack.pop()
            continue
            
        
        if s[i] == ")" and (not stack):
            rev.append(i)
    
    
    rev += stack
    
    rev = sorted(rev)[::-1]
    # ! 从大到小删除
    if rev:
        for i in rev:
            del res[i]
    
    
    return "".join(res)

```
